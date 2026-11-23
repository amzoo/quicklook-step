pub mod mesh;
pub mod stats;
pub mod surface;
pub mod triangulate;
pub mod curve;
pub mod tessellated;
mod swept;

#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum Error {
    #[error("Could not lower point to 2D for triangulation")]
    CouldNotLower,

    #[error("Could not convert into a Surface")]
    UnknownSurfaceType,

    #[error("Could not convert into a Curve")]
    UnknownCurveType,

    #[error("Surface has a degenerate (non-invertible) coordinate frame")]
    DegenerateSurface,

    #[error("Closed NURBS and b-spline surfaces are not implemented")]
    ClosedSurface,

    #[error("Self-intersecting NURBS and b-spline surfaces are not implemented")]
    SelfIntersectingSurface,

    #[error("Self-intersecting NURBS and b-spline curves are not implemented")]
    SelfIntersectingCurve,

    #[error("Could not triangulate the face's boundary")]
    Triangulation,
}
