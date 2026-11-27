pub mod feedback;
pub mod mrpfeedbacklinearization;

pub use feedback::{to_gain_matrix, Feedback, GainInput, Mat3, FB_MRP_FEEDBACK_LINEARIZED};
pub use mrpfeedbacklinearization::{mrp_feedback_linearized_step, MRPFeedbackLinearized};
