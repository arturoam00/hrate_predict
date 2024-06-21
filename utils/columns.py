class Columns:

    DTIME_COLUMN = "Time"
    ACCELEROMETER_X = "Accelerometer_X"
    ACCELEROMETER_Y = "Accelerometer_Y"
    ACCELEROMETER_Z = "Accelerometer_Z"
    BAROMETER_X = "Barometer_X"
    GYROSCOPE_X = "Gyroscope_X"
    GYROSCOPE_Y = "Gyroscope_Y"
    GYROSCOPE_Z = "Gyroscope_Z"
    LINEAR_ACCELEROMETER_X = "Linear_Accelerometer_X"
    LINEAR_ACCELEROMETER_Y = "Linear_Accelerometer_Y"
    LINEAR_ACCELEROMETER_Z = "Linear_Accelerometer_Z"
    LOCATION_LATITUDE = "Location_Latitude"
    LOCATION_LONGITUDE = "Location_Longitude"
    LOCATION_HEIGHT = "Location_Height"
    LOCATION_VELOCITY = "Location_Velocity"
    MAGNETOMETER_X = "Magnetometer_X"
    MAGNETOMETER_Y = "Magnetometer_Y"
    MAGNETOMETER_Z = "Magnetometer_Z"
    PROXIMITY_DISTANCE = "Proximity_Distance"
    HEART_RATE = "Heart_rate_Avg"

    @classmethod
    def get_feature_columns(cls) -> list[str]:
        return [
            cls.ACCELEROMETER_X,
            cls.ACCELEROMETER_Y,
            cls.ACCELEROMETER_Z,
            cls.BAROMETER_X,
            cls.GYROSCOPE_X,
            cls.GYROSCOPE_Y,
            cls.GYROSCOPE_Z,
            cls.LINEAR_ACCELEROMETER_X,
            cls.LINEAR_ACCELEROMETER_Y,
            cls.LINEAR_ACCELEROMETER_Z,
            cls.LOCATION_LATITUDE,
            cls.LOCATION_LONGITUDE,
            cls.LOCATION_HEIGHT,
            cls.LOCATION_VELOCITY,
            cls.MAGNETOMETER_X,
            cls.MAGNETOMETER_Y,
            cls.MAGNETOMETER_Z,
        ]

    @classmethod
    def get_target_column(cls) -> str:
        return cls.HEART_RATE

    @classmethod
    def get_datetime_column(cls) -> str:
        return cls.DTIME_COLUMN
