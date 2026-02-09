use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const LOCATIONS: &[&str] = &[
    "kitchen", "garden", "bedroom", "office", "library", "park", "store", "bathroom", "garage",
    "basement",
];

const NAMES: &[&str] = &[
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank", "Iris", "Jack",
];

const OBJECTS: &[&str] = &[
    "key", "book", "phone", "letter", "map", "ring", "coin", "hat", "bag", "card",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quiz {
    pub question_tokens: Vec<u32>,
    pub answer_idx: usize,
    pub quiz_type: QuizType,
    pub distance: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuizType {
    EntityLocation,
    ObjectHolder,
    StateTracking,
    Contradiction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub chunks: Vec<Vec<u32>>,
    pub quizzes: Vec<(usize, Quiz)>,
}

pub struct Vocabulary {
    pub word_to_id: HashMap<String, u32>,
    pub id_to_word: HashMap<u32, String>,
    next_id: u32,
}

impl Vocabulary {
    pub fn new() -> Self {
        let mut vocab = Vocabulary {
            word_to_id: HashMap::new(),
            id_to_word: HashMap::new(),
            next_id: 1,
        };
        vocab.add_word("<pad>");
        vocab.add_word("<quiz>");
        vocab.add_word("<sep>");
        for name in NAMES {
            vocab.add_word(name);
        }
        for loc in LOCATIONS {
            vocab.add_word(loc);
        }
        for obj in OBJECTS {
            vocab.add_word(obj);
        }
        let filler_words = [
            "the", "a", "an", "is", "in", "at", "to", "went", "moved", "walked", "picked", "up",
            "put", "down", "gave", "took", "from", "and", "then", "after", "that", "later", "next",
            "while", "before", "has", "had", "was", "with", "where", "what", "who", "holds",
            "carrying", "left", "dropped", "found", "lost", "happy", "sad", "alive", "not", "dead",
            "said", "told", "asked", "knows", "thinks", "believes", "saw",
        ];
        for w in &filler_words {
            vocab.add_word(w);
        }
        vocab
    }

    pub fn add_word(&mut self, word: &str) -> u32 {
        if let Some(&id) = self.word_to_id.get(word) {
            return id;
        }
        let id = self.next_id;
        self.word_to_id.insert(word.to_string(), id);
        self.id_to_word.insert(id, word.to_string());
        self.next_id += 1;
        id
    }

    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|w| {
                let lower = w.to_lowercase();
                if let Some(&id) = self.word_to_id.get(&lower) {
                    id
                } else if let Some(&id) = self.word_to_id.get(w) {
                    id
                } else {
                    self.add_word(&lower)
                }
            })
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|id| {
                self.id_to_word
                    .get(id)
                    .map(|s| s.as_str())
                    .unwrap_or("<unk>")
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn size(&self) -> usize {
        self.next_id as usize
    }

    pub fn get_id(&self, word: &str) -> Option<u32> {
        self.word_to_id.get(word).copied()
    }
}

pub struct EntityTrackingGenerator {
    rng: ChaCha8Rng,
}

impl EntityTrackingGenerator {
    pub fn new(seed: u64) -> Self {
        EntityTrackingGenerator {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    pub fn generate(&mut self, vocab: &mut Vocabulary, num_chunks: usize) -> TrainingExample {
        let num_entities = self.rng.gen_range(3..=6).min(NAMES.len());
        let names: Vec<&str> = NAMES.iter().copied().take(num_entities).collect();

        let mut entity_locations: HashMap<String, String> = HashMap::new();
        let mut entity_objects: HashMap<String, Vec<String>> = HashMap::new();

        for name in &names {
            let loc = LOCATIONS[self.rng.gen_range(0..LOCATIONS.len())];
            entity_locations.insert(name.to_string(), loc.to_string());
            entity_objects.insert(name.to_string(), Vec::new());
        }

        let num_objects = self.rng.gen_range(2..=4).min(OBJECTS.len());
        for i in 0..num_objects {
            let holder = &names[self.rng.gen_range(0..names.len())];
            entity_objects
                .get_mut(*holder)
                .unwrap()
                .push(OBJECTS[i].to_string());
        }

        let mut chunks = Vec::with_capacity(num_chunks);
        let mut quizzes = Vec::new();

        for chunk_idx in 0..num_chunks {
            let mut sentences = Vec::new();

            let num_events = self.rng.gen_range(2..=4);
            for _ in 0..num_events {
                let event_type = self.rng.gen_range(0..3);
                let entity_idx = self.rng.gen_range(0..names.len());
                let entity = names[entity_idx];

                let sentence = match event_type {
                    0 => {
                        let new_loc = LOCATIONS[self.rng.gen_range(0..LOCATIONS.len())];
                        entity_locations.insert(entity.to_string(), new_loc.to_string());
                        format!("{} went to the {}", entity, new_loc)
                    }
                    1 => {
                        if !entity_objects[entity].is_empty() && self.rng.gen_bool(0.5) {
                            let obj_idx = self.rng.gen_range(0..entity_objects[entity].len());
                            let obj = entity_objects.get_mut(entity).unwrap().remove(obj_idx);
                            format!("{} dropped the {}", entity, obj)
                        } else {
                            let available: Vec<&str> = OBJECTS
                                .iter()
                                .copied()
                                .filter(|o| {
                                    !entity_objects.values().any(|v| v.contains(&o.to_string()))
                                })
                                .collect();
                            if !available.is_empty() {
                                let obj = available[self.rng.gen_range(0..available.len())];
                                entity_objects
                                    .get_mut(entity)
                                    .unwrap()
                                    .push(obj.to_string());
                                format!("{} picked up the {}", entity, obj)
                            } else {
                                let loc = &entity_locations[entity];
                                format!("{} is in the {}", entity, loc)
                            }
                        }
                    }
                    _ => {
                        if names.len() > 1 {
                            let mut other_idx = self.rng.gen_range(0..names.len());
                            while other_idx == entity_idx {
                                other_idx = self.rng.gen_range(0..names.len());
                            }
                            let other = names[other_idx];
                            if !entity_objects[entity].is_empty() && self.rng.gen_bool(0.3) {
                                let obj_idx = self.rng.gen_range(0..entity_objects[entity].len());
                                let obj = entity_objects.get_mut(entity).unwrap().remove(obj_idx);
                                entity_objects.get_mut(other).unwrap().push(obj.clone());
                                format!("{} gave the {} to {}", entity, obj, other)
                            } else {
                                let loc = &entity_locations[entity];
                                format!("{} saw {} in the {}", entity, other, loc)
                            }
                        } else {
                            let loc = &entity_locations[entity];
                            format!("{} waited in the {}", entity, loc)
                        }
                    }
                };
                sentences.push(sentence);
            }

            let chunk_text = sentences.join(" then ");
            let chunk_tokens = vocab.encode(&chunk_text);
            chunks.push(chunk_tokens);

            if chunk_idx > 0 && self.rng.gen_bool(0.7) {
                let quiz_entity = names[self.rng.gen_range(0..names.len())];
                let quiz_type = if self.rng.gen_bool(0.6) {
                    QuizType::EntityLocation
                } else {
                    QuizType::ObjectHolder
                };

                match quiz_type {
                    QuizType::EntityLocation => {
                        let question = format!("where is {}", quiz_entity);
                        let question_tokens = vocab.encode(&question);
                        let correct_loc = &entity_locations[quiz_entity];
                        let answer_idx =
                            LOCATIONS.iter().position(|l| l == correct_loc).unwrap_or(0);
                        quizzes.push((
                            chunk_idx,
                            Quiz {
                                question_tokens,
                                answer_idx,
                                quiz_type: QuizType::EntityLocation,
                                distance: chunk_idx,
                            },
                        ));
                    }
                    QuizType::ObjectHolder => {
                        let objs = &entity_objects[quiz_entity];
                        if !objs.is_empty() {
                            let obj = &objs[0];
                            let question = format!("who holds the {}", obj);
                            let question_tokens = vocab.encode(&question);
                            let answer_idx =
                                names.iter().position(|n| *n == quiz_entity).unwrap_or(0);
                            quizzes.push((
                                chunk_idx,
                                Quiz {
                                    question_tokens,
                                    answer_idx,
                                    quiz_type: QuizType::ObjectHolder,
                                    distance: chunk_idx,
                                },
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }

        TrainingExample { chunks, quizzes }
    }

    pub fn generate_batch(
        &mut self,
        vocab: &mut Vocabulary,
        batch_size: usize,
        num_chunks: usize,
    ) -> Vec<TrainingExample> {
        (0..batch_size)
            .map(|_| self.generate(vocab, num_chunks))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary() {
        let mut vocab = Vocabulary::new();
        let tokens = vocab.encode("Alice went to the kitchen");
        assert!(!tokens.is_empty());
        let decoded = vocab.decode(&tokens);
        assert_eq!(decoded, "Alice went to the kitchen");
    }

    #[test]
    fn test_generation() {
        let mut vocab = Vocabulary::new();
        let mut gen = EntityTrackingGenerator::new(42);
        let example = gen.generate(&mut vocab, 5);
        assert_eq!(example.chunks.len(), 5);
        assert!(!example.chunks[0].is_empty());
    }

    #[test]
    fn test_quiz_generation() {
        let mut vocab = Vocabulary::new();
        let mut gen = EntityTrackingGenerator::new(42);
        let example = gen.generate(&mut vocab, 10);
        assert!(
            !example.quizzes.is_empty(),
            "should generate quizzes for 10 chunks"
        );
        for (chunk_idx, quiz) in &example.quizzes {
            assert!(*chunk_idx > 0);
            assert!(!quiz.question_tokens.is_empty());
        }
    }
}
