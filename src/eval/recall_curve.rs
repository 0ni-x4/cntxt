use std::collections::BTreeMap;

pub struct RecallTracker {
    results: BTreeMap<usize, (usize, usize)>,
}

impl RecallTracker {
    pub fn new() -> Self {
        RecallTracker {
            results: BTreeMap::new(),
        }
    }

    pub fn record(&mut self, distance: usize, correct: bool) {
        let entry = self.results.entry(distance).or_insert((0, 0));
        entry.0 += 1;
        if correct {
            entry.1 += 1;
        }
    }

    pub fn get_curve(&self) -> Vec<(usize, f64)> {
        self.results
            .iter()
            .map(|(dist, (total, correct))| {
                let accuracy = if *total > 0 {
                    *correct as f64 / *total as f64
                } else {
                    0.0
                };
                (*dist, accuracy)
            })
            .collect()
    }

    pub fn reset(&mut self) {
        self.results.clear();
    }

    pub fn total_quizzes(&self) -> usize {
        self.results.values().map(|(t, _)| t).sum()
    }

    pub fn overall_accuracy(&self) -> f64 {
        let total: usize = self.results.values().map(|(t, _)| t).sum();
        let correct: usize = self.results.values().map(|(_, c)| c).sum();
        if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_tracker() {
        let mut tracker = RecallTracker::new();
        tracker.record(1, true);
        tracker.record(1, true);
        tracker.record(1, false);
        tracker.record(5, true);
        tracker.record(5, false);
        tracker.record(5, false);

        let curve = tracker.get_curve();
        assert_eq!(curve.len(), 2);

        let d1 = curve.iter().find(|(d, _)| *d == 1).unwrap();
        assert!((d1.1 - 0.6667).abs() < 0.01);

        let d5 = curve.iter().find(|(d, _)| *d == 5).unwrap();
        assert!((d5.1 - 0.3333).abs() < 0.01);
    }
}
