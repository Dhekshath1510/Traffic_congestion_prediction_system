from datetime import datetime
from pydantic import BaseModel, Field


class TrafficRecord(BaseModel):
    """Canonical representation of one traffic observation."""

    timestamp: datetime = Field(
        description="Timestamp of the traffic observation.",
    )

    road_segment_id: str = Field(
        min_length=1,
        description="Unique identifier of the road segment.",
    )

    traffic_flow: float = Field(
        ge=0,
        description="Traffic flow in vehicles per hour.",
    )

    average_speed: float | None = Field(
        default=None,
        ge=0,
        description="Average vehicle speed in km/h.",
    )

    vehicle_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of vehicles observed in the interval.",
    )

    traffic_density: float | None = Field(
        default=None,
        ge=0,
        description="Traffic density, when available.",
    )

    occupancy: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Road occupancy percentage, when available.",
    )
