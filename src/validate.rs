//! Validation utilities for fusion results.
//!
//! This module provides functions to validate fusion results, ensuring they meet
//! expected properties (e.g., sorted by score, no duplicates, valid scores).

use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

/// Validation result for a fusion output.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationResult {
    /// Whether validation passed.
    pub is_valid: bool,
    /// List of validation errors (if any).
    pub errors: Vec<String>,
    /// List of warnings (non-critical issues).
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create a valid result.
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Create an invalid result with errors.
    pub fn invalid(errors: Vec<String>) -> Self {
        Self {
            is_valid: false,
            errors,
            warnings: Vec::new(),
        }
    }

    /// Add a warning.
    pub fn with_warning(mut self, warning: String) -> Self {
        self.warnings.push(warning);
        self
    }
}

/// Validate that fusion results are properly sorted by score (descending).
///
/// # Returns
/// `ValidationResult` indicating whether results are sorted, with errors if not.
pub fn validate_sorted<I: Clone + Eq + Hash + Debug>(results: &[(I, f32)]) -> ValidationResult {
    if results.is_empty() {
        return ValidationResult::valid();
    }

    let mut errors = Vec::new();
    for (i, window) in results.windows(2).enumerate() {
        let (_, score_a) = &window[0];
        let (_, score_b) = &window[1];
        if score_a < score_b {
            errors.push(format!(
                "Results not sorted: position {} has score {} < position {} has score {}",
                i,
                score_a,
                i + 1,
                score_b
            ));
        }
    }

    if errors.is_empty() {
        ValidationResult::valid()
    } else {
        ValidationResult::invalid(errors)
    }
}

/// Validate that fusion results contain no duplicate document IDs.
///
/// # Returns
/// `ValidationResult` indicating whether results are unique, with errors if duplicates found.
pub fn validate_no_duplicates<I: Clone + Eq + Hash + Debug>(
    results: &[(I, f32)],
) -> ValidationResult {
    let mut seen = HashSet::new();
    let mut errors = Vec::new();

    for (i, (id, _)) in results.iter().enumerate() {
        if seen.contains(id) {
            // Format ID as string (works for any type that implements Display or Debug)
            let id_str = format!("{:?}", id);
            errors.push(format!(
                "Duplicate document ID at position {}: {}",
                i, id_str
            ));
        } else {
            seen.insert(id.clone());
        }
    }

    if errors.is_empty() {
        ValidationResult::valid()
    } else {
        ValidationResult::invalid(errors)
    }
}

/// Validate that all scores are finite (not NaN or Infinity).
///
/// # Returns
/// `ValidationResult` indicating whether all scores are finite, with errors if not.
pub fn validate_finite_scores<I: Clone + Eq + Hash + Debug>(
    results: &[(I, f32)],
) -> ValidationResult {
    let mut errors = Vec::new();

    for (i, (id, score)) in results.iter().enumerate() {
        if !score.is_finite() {
            let id_str = format!("{:?}", id);
            errors.push(format!(
                "Non-finite score at position {} for document {}: {}",
                i, id_str, score
            ));
        }
    }

    if errors.is_empty() {
        ValidationResult::valid()
    } else {
        ValidationResult::invalid(errors)
    }
}

/// Validate that all scores are non-negative.
///
/// # Returns
/// `ValidationResult` indicating whether all scores are non-negative, with warnings for negative scores.
pub fn validate_non_negative_scores<I: Clone + Eq + Hash + Debug>(
    results: &[(I, f32)],
) -> ValidationResult {
    let mut warnings = Vec::new();

    for (i, (id, score)) in results.iter().enumerate() {
        if *score < 0.0 {
            let id_str = format!("{:?}", id);
            warnings.push(format!(
                "Negative score at position {} for document {}: {}",
                i, id_str, score
            ));
        }
    }

    if warnings.is_empty() {
        ValidationResult::valid()
    } else {
        ValidationResult::valid().with_warning(format!(
            "Found {} negative scores (may be expected for some algorithms)",
            warnings.len()
        ))
    }
}

/// Validate that results are within expected bounds (e.g., top_k).
///
/// # Arguments
/// * `results` - Fusion results to validate
/// * `max_results` - Maximum expected number of results (None = no limit)
///
/// # Returns
/// `ValidationResult` indicating whether results are within bounds, with warnings if exceeded.
pub fn validate_bounds<I: Clone + Eq + Hash + Debug>(
    results: &[(I, f32)],
    max_results: Option<usize>,
) -> ValidationResult {
    if let Some(max) = max_results {
        if results.len() > max {
            return ValidationResult::valid().with_warning(format!(
                "Results exceed expected maximum: {} > {}",
                results.len(),
                max
            ));
        }
    }
    ValidationResult::valid()
}

/// Comprehensive validation of fusion results.
///
/// Performs all validation checks:
/// - Sorted by score (descending)
/// - No duplicate document IDs
/// - All scores are finite
/// - Optional: non-negative scores (warning only)
/// - Optional: within bounds (warning only)
///
/// # Arguments
/// * `results` - Fusion results to validate
/// * `check_non_negative` - Whether to warn on negative scores (default: false)
/// * `max_results` - Maximum expected number of results (None = no limit)
///
/// # Returns
/// `ValidationResult` with all errors and warnings.
pub fn validate<I: Clone + Eq + Hash + Debug>(
    results: &[(I, f32)],
    check_non_negative: bool,
    max_results: Option<usize>,
) -> ValidationResult {
    let mut all_errors = Vec::new();
    let mut all_warnings = Vec::new();

    // Check sorted
    let sorted_result = validate_sorted(results);
    if !sorted_result.is_valid {
        all_errors.extend(sorted_result.errors);
    }

    // Check duplicates
    let dup_result = validate_no_duplicates(results);
    if !dup_result.is_valid {
        all_errors.extend(dup_result.errors);
    }

    // Check finite scores
    let finite_result = validate_finite_scores(results);
    if !finite_result.is_valid {
        all_errors.extend(finite_result.errors);
    }

    // Optional: check non-negative (warning only)
    if check_non_negative {
        let non_neg_result = validate_non_negative_scores(results);
        all_warnings.extend(non_neg_result.warnings);
    }

    // Optional: check bounds (warning only)
    let bounds_result = validate_bounds(results, max_results);
    all_warnings.extend(bounds_result.warnings);

    if all_errors.is_empty() {
        ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: all_warnings,
        }
    } else {
        ValidationResult {
            is_valid: false,
            errors: all_errors,
            warnings: all_warnings,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_sorted() {
        let sorted = vec![("a", 0.9), ("b", 0.8), ("c", 0.7)];
        assert!(validate_sorted(&sorted).is_valid);

        let unsorted = vec![("a", 0.9), ("b", 0.95), ("c", 0.7)];
        assert!(!validate_sorted(&unsorted).is_valid);
    }

    #[test]
    fn test_validate_no_duplicates() {
        let unique = vec![("a", 0.9), ("b", 0.8), ("c", 0.7)];
        assert!(validate_no_duplicates(&unique).is_valid);

        let duplicates = vec![("a", 0.9), ("b", 0.8), ("a", 0.7)];
        assert!(!validate_no_duplicates(&duplicates).is_valid);
    }

    #[test]
    fn test_validate_finite_scores() {
        let finite = vec![("a", 0.9), ("b", 0.8)];
        assert!(validate_finite_scores(&finite).is_valid);

        let nan = vec![("a", 0.9), ("b", f32::NAN)];
        assert!(!validate_finite_scores(&nan).is_valid);

        let inf = vec![("a", 0.9), ("b", f32::INFINITY)];
        assert!(!validate_finite_scores(&inf).is_valid);
    }

    #[test]
    fn test_validate_comprehensive() {
        let valid = vec![("a", 0.9), ("b", 0.8), ("c", 0.7)];
        let result = validate(&valid, false, None);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());

        let invalid = vec![("a", 0.9), ("b", 0.95), ("a", 0.7)]; // Unsorted + duplicate
        let result = validate(&invalid, false, None);
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }
}
