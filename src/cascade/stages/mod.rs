//! Built-in classification stages.
//!
//! This module contains the default stages used by the cascade classifier:
//!
//! - `ExtensionStage`: Analyzes file extensions and size distribution
//! - `PatternStage`: Matches patterns in torrent names
//! - `MlStage`: ML fallback for ambiguous cases

mod extensions;
mod ml;
mod patterns;

pub use extensions::ExtensionStage;
pub use ml::MlStage;
pub use patterns::PatternStage;
