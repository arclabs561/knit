//! WebAssembly bindings for `clasp`.
//!
//! This crate exists to keep `clasp` itself small and dependency-light.

use wasm_bindgen::prelude::*;

use clasp::{
    additive_multi_task_with_config, standardized_with_config, AdditiveMultiTaskConfig,
    FusionConfig, Normalization, RrfConfig, StandardizedConfig, WeightedConfig,
};

/// Helper to convert JS array of [id, score] pairs to Vec<(String, f32)>.
fn js_to_results(js: &JsValue) -> Result<Vec<(String, f32)>, JsValue> {
    use wasm_bindgen::JsCast;

    let array = js
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array"))?;

    let len = array.length() as usize;
    let mut results = Vec::with_capacity(len);

    for (idx, item) in array.iter().enumerate() {
        let pair = item.dyn_ref::<js_sys::Array>().ok_or_else(|| {
            JsValue::from_str(&format!("Expected [id, score] pair at index {}", idx))
        })?;
        if pair.length() != 2 {
            return Err(JsValue::from_str(&format!(
                "Expected [id, score] pair at index {}, got array of length {}",
                idx,
                pair.length()
            )));
        }
        let id = pair
            .get(0)
            .as_string()
            .ok_or_else(|| JsValue::from_str(&format!("id must be a string at index {}", idx)))?;
        let score_val = pair.get(1).as_f64().ok_or_else(|| {
            JsValue::from_str(&format!("score must be a number at index {}", idx))
        })?;

        if !score_val.is_finite() {
            return Err(JsValue::from_str(&format!(
                "score must be a finite number at index {}, got {}",
                idx, score_val
            )));
        }

        results.push((id, score_val as f32));
    }
    Ok(results)
}

fn results_to_js(results: &[(String, f32)]) -> JsValue {
    let array = js_sys::Array::new();
    for (id, score) in results {
        let pair = js_sys::Array::new();
        pair.push(&JsValue::from_str(id));
        pair.push(&JsValue::from_f64(*score as f64));
        array.push(&pair);
    }
    array.into()
}

#[wasm_bindgen]
pub fn rrf(
    results_a: &JsValue,
    results_b: &JsValue,
    k: Option<u32>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;

    let k_val = k.unwrap_or(60);
    if k_val == 0 {
        return Err(JsValue::from_str(
            "k must be >= 1 to avoid division by zero",
        ));
    }

    let config = RrfConfig { k: k_val, top_k };
    Ok(results_to_js(&clasp::rrf_with_config(&a, &b, config)))
}

#[wasm_bindgen]
pub fn isr(
    results_a: &JsValue,
    results_b: &JsValue,
    k: Option<u32>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;

    let k_val = k.unwrap_or(1);
    if k_val == 0 {
        return Err(JsValue::from_str(
            "k must be >= 1 to avoid division by zero",
        ));
    }

    let config = RrfConfig { k: k_val, top_k };
    Ok(results_to_js(&clasp::isr_with_config(&a, &b, config)))
}

#[wasm_bindgen]
pub fn combsum(results_a: &JsValue, results_b: &JsValue) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;
    Ok(results_to_js(&clasp::combsum(&a, &b)))
}

#[wasm_bindgen]
pub fn combmnz(results_a: &JsValue, results_b: &JsValue) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;
    Ok(results_to_js(&clasp::combmnz(&a, &b)))
}

#[wasm_bindgen]
pub fn borda(results_a: &JsValue, results_b: &JsValue) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;
    Ok(results_to_js(&clasp::borda(&a, &b)))
}

#[wasm_bindgen]
pub fn dbsf(results_a: &JsValue, results_b: &JsValue) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;
    Ok(results_to_js(&clasp::dbsf(&a, &b)))
}

#[wasm_bindgen]
pub fn weighted(
    results_a: &JsValue,
    results_b: &JsValue,
    weight_a: f32,
    weight_b: f32,
    normalize: Option<bool>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;

    if !weight_a.is_finite() || !weight_b.is_finite() {
        return Err(JsValue::from_str("weights must be finite"));
    }

    let normalize = normalize.unwrap_or(true);
    let config = WeightedConfig {
        weight_a,
        weight_b,
        normalize,
        top_k,
    };
    Ok(results_to_js(&clasp::weighted(&a, &b, config)))
}

#[wasm_bindgen]
pub fn standardized(
    results_a: &JsValue,
    results_b: &JsValue,
    clip_min: Option<f32>,
    clip_max: Option<f32>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;

    let clip_min = clip_min.unwrap_or(-3.0);
    let clip_max = clip_max.unwrap_or(3.0);

    let config = StandardizedConfig {
        clip_range: (clip_min, clip_max),
        top_k,
    };
    Ok(results_to_js(&standardized_with_config(&a, &b, config)))
}

#[wasm_bindgen]
pub fn additive_multi_task(
    results_a: &JsValue,
    results_b: &JsValue,
    weight_a: Option<f32>,
    weight_b: Option<f32>,
    normalization: Option<String>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;

    let normalization = normalization.unwrap_or_else(|| "zscore".to_string());
    let normalization = match normalization.to_lowercase().as_str() {
        "none" => Normalization::None,
        "minmax" => Normalization::MinMax,
        "zscore" => Normalization::ZScore,
        other => {
            return Err(JsValue::from_str(&format!(
                "unknown normalization: {other} (expected: none|minmax|zscore)"
            )))
        }
    };

    let config = AdditiveMultiTaskConfig {
        weights: (weight_a.unwrap_or(1.0), weight_b.unwrap_or(1.0)),
        normalization,
        top_k,
    };

    Ok(results_to_js(&additive_multi_task_with_config(
        &a, &b, config,
    )))
}

#[wasm_bindgen]
pub fn rrf_multi(
    lists: &JsValue,
    k: Option<u32>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    use wasm_bindgen::JsCast;

    let array = lists
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of lists"))?;
    if array.length() == 0 {
        return Ok(js_sys::Array::new().into());
    }

    let mut rust_lists: Vec<Vec<(String, f32)>> = Vec::with_capacity(array.length() as usize);
    for (idx, item) in array.iter().enumerate() {
        let list = item
            .dyn_ref::<js_sys::Array>()
            .ok_or_else(|| JsValue::from_str(&format!("Expected list (array) at index {}", idx)))?;
        rust_lists.push(js_to_results(&list.into())?);
    }

    let k_val = k.unwrap_or(60);
    if k_val == 0 {
        return Err(JsValue::from_str(
            "k must be >= 1 to avoid division by zero",
        ));
    }
    let config = RrfConfig { k: k_val, top_k };
    Ok(results_to_js(&clasp::rrf_multi(&rust_lists, config)))
}

#[wasm_bindgen]
pub fn combsum_multi(lists: &JsValue, top_k: Option<usize>) -> Result<JsValue, JsValue> {
    use wasm_bindgen::JsCast;

    let array = lists
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of lists"))?;
    if array.length() == 0 {
        return Ok(js_sys::Array::new().into());
    }

    let mut rust_lists: Vec<Vec<(String, f32)>> = Vec::with_capacity(array.length() as usize);
    for (idx, item) in array.iter().enumerate() {
        let list = item
            .dyn_ref::<js_sys::Array>()
            .ok_or_else(|| JsValue::from_str(&format!("Expected list (array) at index {}", idx)))?;
        rust_lists.push(js_to_results(&list.into())?);
    }

    let config = FusionConfig { top_k };
    Ok(results_to_js(&clasp::combsum_multi(&rust_lists, config)))
}
