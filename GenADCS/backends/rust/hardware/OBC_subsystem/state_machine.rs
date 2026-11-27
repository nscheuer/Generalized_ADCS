use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use adcs_subsystem::goals::Goal;

pub type HardwareMaskSpec = BTreeMap<String, bool>;

#[derive(Clone)]
pub enum ADCSBinding {
    ModeName(String),
    GoalOnly(Arc<dyn Goal + Send + Sync>),
    ModeAndGoal(String, Arc<dyn Goal + Send + Sync>),
}

#[derive(Clone)]
pub enum Trigger {
    Telecommand(Telecommand),
    InFOV(InFOV),
    Completion(Completion),
    After(After),
    FDIRTriggered(FDIRTriggered),
}

#[derive(Clone)]
pub struct Telecommand {
    pub command: String,
    pub goal: Option<Arc<dyn Goal + Send + Sync>>,
    pub sensors: Option<HardwareMaskSpec>,
    pub actuators: Option<HardwareMaskSpec>,
}

impl Telecommand {
    pub fn new(
        command: String,
        goal: Option<Arc<dyn Goal + Send + Sync>>,
        sensors: Option<HardwareMaskSpec>,
        actuators: Option<HardwareMaskSpec>,
    ) -> Result<Self, String> {
        if command.is_empty() {
            return Err("telecommand name must be nonempty".into());
        }
        Ok(Self {
            command,
            goal,
            sensors,
            actuators,
        })
    }
}

#[derive(Clone)]
pub struct InFOV {
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
    pub max_off_nadir_deg: f64,
    pub negate: bool,
    pub goal: Option<Arc<dyn Goal + Send + Sync>>,
}

impl InFOV {
    pub fn new(
        lat: f64,
        lon: f64,
        alt: f64,
        max_off_nadir_deg: f64,
        negate: bool,
        goal: Option<Arc<dyn Goal + Send + Sync>>,
    ) -> Result<Self, String> {
        if !lat.is_finite() || !lon.is_finite() || !alt.is_finite() || !max_off_nadir_deg.is_finite()
        {
            return Err("InFOV parameters must be finite".into());
        }
        if !(-90.0..=90.0).contains(&lat) {
            return Err("latitude must be between -90 and 90 degrees".into());
        }
        if !(-180.0..=180.0).contains(&lon) {
            return Err("longitude must be between -180 and 180 degrees".into());
        }
        if max_off_nadir_deg <= 0.0 || max_off_nadir_deg > 90.0 {
            return Err("max_off_nadir_deg must be in (0, 90]".into());
        }
        Ok(Self {
            lat,
            lon,
            alt,
            max_off_nadir_deg,
            negate,
            goal,
        })
    }

    pub fn inverted(&self) -> Self {
        Self {
            lat: self.lat,
            lon: self.lon,
            alt: self.alt,
            max_off_nadir_deg: self.max_off_nadir_deg,
            negate: !self.negate,
            goal: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Completion;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct After {
    pub duration: f64,
}

impl After {
    pub fn new(duration: f64) -> Result<Self, String> {
        if !duration.is_finite() || duration <= 0.0 {
            return Err("duration must be finite and positive seconds".into());
        }
        Ok(Self { duration })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FDIRTriggered {
    pub monitor: Option<String>,
}

#[derive(Clone, Default)]
pub struct ModeBindings {
    pub adcs: Option<ADCSBinding>,
    pub sensors: Option<HardwareMaskSpec>,
    pub actuators: Option<HardwareMaskSpec>,
}

#[derive(Clone)]
pub struct Mode {
    pub name: String,
    pub submodes: Vec<Mode>,
    pub initial_submode: Option<String>,
    pub duration: Option<f64>,
    pub auto_target: Option<String>,
    pub bindings: ModeBindings,
}

impl Mode {
    pub fn new(
        name: String,
        submodes: Vec<Mode>,
        initial_submode: Option<String>,
        duration: Option<f64>,
        auto_target: Option<String>,
        bindings: ModeBindings,
    ) -> Result<Self, String> {
        if name.is_empty() || name.contains('.') || name.contains('*') {
            return Err(format!("invalid mode name: {name:?}"));
        }
        if let Some(d) = duration {
            if !d.is_finite() || d <= 0.0 {
                return Err("mode duration must be finite and positive seconds".into());
            }
        }
        let mut init_sub = initial_submode;
        if !submodes.is_empty() {
            let mut seen = HashSet::new();
            for sub in &submodes {
                if !seen.insert(sub.name.clone()) {
                    return Err(format!("duplicate submode names in {name:?}"));
                }
            }
            if init_sub.is_none() {
                init_sub = Some(submodes[0].name.clone());
            } else if let Some(ref s) = init_sub {
                if !seen.contains(s) {
                    return Err(format!("initial_submode {s:?} not in {name:?}"));
                }
            }
        } else if init_sub.is_some() {
            return Err("initial_submode requires submodes".into());
        }

        Ok(Self {
            name,
            submodes,
            initial_submode: init_sub,
            duration,
            auto_target,
            bindings,
        })
    }
}

#[derive(Clone)]
pub struct Transition {
    pub source: String,
    pub target: String,
    pub trigger: Trigger,
    pub priority: i32,
}

impl Transition {
    pub fn new(
        source: String,
        target: String,
        trigger: Trigger,
        priority: i32,
    ) -> Result<Self, String> {
        if source.is_empty() || target.is_empty() {
            return Err("transition source and target must be nonempty".into());
        }
        Ok(Self {
            source,
            target,
            trigger,
            priority,
        })
    }
}

pub type TimelineEntry = (f64, String, Option<Arc<dyn Goal + Send + Sync>>);

#[derive(Clone)]
pub struct StateMachine {
    pub initial: String,
    pub modes: Vec<Mode>,
    pub transitions: Vec<Transition>,
    pub timeline: Vec<TimelineEntry>,
}

impl StateMachine {
    pub fn new(
        initial: String,
        modes: Vec<Mode>,
        transitions: Vec<Transition>,
        timeline: Vec<TimelineEntry>,
    ) -> Result<Self, String> {
        if modes.is_empty() {
            return Err("StateMachine requires at least one Mode".into());
        }
        let mut seen = HashSet::new();
        for m in &modes {
            if !seen.insert(m.name.clone()) {
                return Err("duplicate top-level mode names".into());
            }
        }
        if initial.is_empty() {
            return Err("initial mode must be nonempty".into());
        }
        for (t, cmd, _) in &timeline {
            if !t.is_finite() || *t < 0.0 {
                return Err("timeline times must be finite and nonnegative".into());
            }
            if cmd.is_empty() {
                return Err("timeline telecommand names must be nonempty".into());
            }
        }
        Ok(Self {
            initial,
            modes,
            transitions,
            timeline,
        })
    }
}
