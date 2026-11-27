use std::collections::HashSet;
use std::sync::Arc;

pub trait FDIRMonitor: Send + Sync {
    fn name(&self) -> &str;
}

#[derive(Clone, Debug, PartialEq)]
pub struct AngularRateMonitor {
    pub max_rate: f64,
    pub name: String,
    pub target_mode: Option<String>,
}

impl AngularRateMonitor {
    pub fn new(
        max_rate: f64,
        name: Option<String>,
        target_mode: Option<String>,
    ) -> Result<Self, String> {
        if !max_rate.is_finite() || max_rate <= 0.0 {
            return Err("max_rate must be finite and positive rad/s".into());
        }
        let name = name.unwrap_or_else(|| "angular_rate".to_string());
        if name.is_empty() {
            return Err("monitor name must be nonempty".into());
        }
        Ok(Self {
            max_rate,
            name,
            target_mode,
        })
    }
}

impl FDIRMonitor for AngularRateMonitor {
    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Clone)]
pub struct FDIR {
    pub monitors: Vec<Arc<dyn FDIRMonitor>>,
    pub safe_mode: String,
    pub angular_rate_monitors: Vec<AngularRateMonitor>,
}

impl FDIR {
    pub fn new(
        monitors: Vec<AngularRateMonitor>,
        safe_mode: Option<String>,
    ) -> Result<Self, String> {
        let safe_mode = safe_mode.unwrap_or_else(|| "safe_mode".to_string());
        if safe_mode.is_empty() {
            return Err("safe_mode must be a nonempty mode name".into());
        }
        let mut seen = HashSet::new();
        let mut trait_monitors: Vec<Arc<dyn FDIRMonitor>> = Vec::with_capacity(monitors.len());
        for mon in &monitors {
            if !seen.insert(mon.name.clone()) {
                return Err(format!("duplicate FDIR monitor name: {:?}", mon.name));
            }
            trait_monitors.push(Arc::new(mon.clone()));
        }
        Ok(Self {
            monitors: trait_monitors,
            safe_mode,
            angular_rate_monitors: monitors,
        })
    }
}
