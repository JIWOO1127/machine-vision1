"""
live_map_view.py

사용자가 실측/CAD로 제공한 실내 구조를 간략한 XY 평면도로 렌더링하고,
ArUco로 계산한 현재 카메라 위치와 바라보는 방향을 실시간 표시한다.

단위: mm

표시:
- 검은 선: 벽
- 회색 사각형: 구조물/회의실/테이블 등
- 청록 점: 등록된 ArUco 마커
- 빨간 점: 현재 카메라 위치
- 빨간 화살표: 카메라가 바라보는 XY 방향
- 문 개구부는 벽의 빈 구간으로 표현

주의:
이 지도는 현재까지 제공된 실측값을 기반으로 한 간략 모델이다.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from aruco_world_map import ARUCO_MARKERS


# ============================================================
# 현재까지 제공된 지도 범위
# ============================================================

MAP_X_MIN = -5000.0
MAP_X_MAX = 29000.0
MAP_Y_MIN = -700.0
MAP_Y_MAX = 8200.0


# 직사각형 구조물:
# (name, x1, y1, x2, y2)
RECTANGLES = (
    ("table_1", -940.0, 1962.0, 3620.0, 4280.0),
    ("water_1", 4832.0, 2282.0, 5610.0, 3710.0),
    ("meeting_1", 5610.0, 1962.0, 11250.0, 7370.0),
    ("meeting_2", 12725.0, 1962.0, 18875.0, 7380.0),
    ("table_2", 18875.0, 1962.0, 19475.0, 4262.0),
    ("water_2", 18875.0, 4262.0, 19475.0, 4902.0),
    ("wall_block", 18875.0, 4902.0, 19475.0, 6792.0),
    ("table_3", 21607.0, 5252.0, 23707.0, 6125.0),
)

# y=0 벽의 문 개구부
BOTTOM_DOORS = (
    (0.0, 1000.0, "C4"),
    (9680.0, 10680.0, "C3"),
    (19140.0, 20140.0, "C2"),
)

# y=6792 벽의 쪽문
SIDE_DOOR_X1 = 19475.0
SIDE_DOOR_X2 = 21137.0
SIDE_WALL_Y = 6792.0

# 동쪽 끝 세로벽
EAST_WALL_X = 27660.0


class LiveMapRenderer:
    def __init__(
        self,
        width: int = 1100,
        height: int = 420,
        margin: int = 35,
    ):
        self.width = int(width)
        self.height = int(height)
        self.margin = int(margin)

        self._base = self._build_base_map()

    # --------------------------------------------------------
    # 좌표 변환
    # --------------------------------------------------------

    def world_to_pixel(
        self,
        x_mm: float,
        y_mm: float,
    ) -> Tuple[int, int]:

        usable_w = (
            self.width
            -
            2 * self.margin
        )

        usable_h = (
            self.height
            -
            2 * self.margin
        )

        x_norm = (
            (float(x_mm) - MAP_X_MIN)
            /
            (MAP_X_MAX - MAP_X_MIN)
        )

        y_norm = (
            (float(y_mm) - MAP_Y_MIN)
            /
            (MAP_Y_MAX - MAP_Y_MIN)
        )

        px = (
            self.margin
            +
            x_norm * usable_w
        )

        # OpenCV 화면의 +Y는 아래쪽이므로 뒤집는다.
        py = (
            self.height
            -
            self.margin
            -
            y_norm * usable_h
        )

        return (
            int(round(px)),
            int(round(py)),
        )

    # --------------------------------------------------------
    # 기본 지도
    # --------------------------------------------------------

    def _build_base_map(
        self,
    ) -> np.ndarray:

        image = np.full(
            (
                self.height,
                self.width,
                3,
            ),
            245,
            dtype=np.uint8,
        )

        # 지도 테두리
        cv2.rectangle(
            image,
            (1, 1),
            (
                self.width - 2,
                self.height - 2,
            ),
            (80, 80, 80),
            1,
        )

        # 좌표축 안내
        self._draw_axis_legend(
            image
        )

        # 구조물
        for (
            name,
            x1,
            y1,
            x2,
            y2,
        ) in RECTANGLES:

            p1 = self.world_to_pixel(
                min(x1, x2),
                min(y1, y2),
            )

            p2 = self.world_to_pixel(
                max(x1, x2),
                max(y1, y2),
            )

            left = min(
                p1[0],
                p2[0],
            )
            right = max(
                p1[0],
                p2[0],
            )
            top = min(
                p1[1],
                p2[1],
            )
            bottom = max(
                p1[1],
                p2[1],
            )

            cv2.rectangle(
                image,
                (left, top),
                (right, bottom),
                (210, 210, 210),
                -1,
            )

            cv2.rectangle(
                image,
                (left, top),
                (right, bottom),
                (110, 110, 110),
                1,
            )

            center = (
                int(
                    (left + right) / 2
                ),
                int(
                    (top + bottom) / 2
                ),
            )

            cv2.putText(
                image,
                name,
                (
                    center[0] - 25,
                    center[1],
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )

        # y=0 벽. 문 구간만 비워 둔다.
        self._draw_horizontal_wall_with_openings(
            image=image,
            y_mm=0.0,
            x_min=MAP_X_MIN,
            x_max=MAP_X_MAX,
            openings=[
                (a, b)
                for a, b, _
                in BOTTOM_DOORS
            ],
        )

        # 문 라벨
        for (
            x1,
            x2,
            label,
        ) in BOTTOM_DOORS:
            center_x = (
                x1 + x2
            ) / 2.0

            p = self.world_to_pixel(
                center_x,
                0.0,
            )

            cv2.putText(
                image,
                label,
                (
                    p[0] - 12,
                    p[1] - 8,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (60, 60, 60),
                1,
                cv2.LINE_AA,
            )

        # y=6792 벽
        self._draw_horizontal_wall_with_openings(
            image=image,
            y_mm=SIDE_WALL_Y,
            x_min=MAP_X_MIN,
            x_max=MAP_X_MAX,
            openings=[
                (
                    SIDE_DOOR_X1,
                    SIDE_DOOR_X2,
                )
            ],
        )

        side_center = (
            SIDE_DOOR_X1
            +
            SIDE_DOOR_X2
        ) / 2.0

        side_p = self.world_to_pixel(
            side_center,
            SIDE_WALL_Y,
        )

        cv2.putText(
            image,
            "SIDE",
            (
                side_p[0] - 18,
                side_p[1] - 8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

        # x=27660 세로벽
        top = self.world_to_pixel(
            EAST_WALL_X,
            MAP_Y_MAX,
        )

        bottom = self.world_to_pixel(
            EAST_WALL_X,
            MAP_Y_MIN,
        )

        cv2.line(
            image,
            top,
            bottom,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )

        # 등록된 마커 위치
        for marker in ARUCO_MARKERS:
            p = self.world_to_pixel(
                marker.x,
                marker.y,
            )

            cv2.circle(
                image,
                p,
                4,
                (180, 110, 0),
                -1,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                marker.point_name,
                (
                    p[0] + 5,
                    p[1] - 4,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (120, 80, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            image,
            "Registered ArUco",
            (
                self.margin,
                self.height - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (120, 80, 0),
            1,
            cv2.LINE_AA,
        )

        return image

    def _draw_axis_legend(
        self,
        image: np.ndarray,
    ) -> None:

        origin = (
            self.width - 125,
            self.height - 55,
        )

        # +X right
        cv2.arrowedLine(
            image,
            origin,
            (
                origin[0] + 55,
                origin[1],
            ),
            (60, 60, 60),
            2,
            tipLength=0.2,
        )

        cv2.putText(
            image,
            "+X",
            (
                origin[0] + 60,
                origin[1] + 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

        # +Y up
        cv2.arrowedLine(
            image,
            origin,
            (
                origin[0],
                origin[1] - 42,
            ),
            (60, 60, 60),
            2,
            tipLength=0.2,
        )

        cv2.putText(
            image,
            "+Y",
            (
                origin[0] - 12,
                origin[1] - 48,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

    def _draw_horizontal_wall_with_openings(
        self,
        image: np.ndarray,
        y_mm: float,
        x_min: float,
        x_max: float,
        openings: Sequence[
            Tuple[float, float]
        ],
    ) -> None:

        sorted_openings = sorted(
            (
                min(a, b),
                max(a, b),
            )
            for a, b
            in openings
        )

        current = float(
            x_min
        )

        for (
            opening_start,
            opening_end,
        ) in sorted_openings:

            if current < opening_start:
                p1 = (
                    self.world_to_pixel(
                        current,
                        y_mm,
                    )
                )

                p2 = (
                    self.world_to_pixel(
                        opening_start,
                        y_mm,
                    )
                )

                cv2.line(
                    image,
                    p1,
                    p2,
                    (20, 20, 20),
                    3,
                    cv2.LINE_AA,
                )

            current = max(
                current,
                opening_end,
            )

        if current < x_max:
            p1 = self.world_to_pixel(
                current,
                y_mm,
            )

            p2 = self.world_to_pixel(
                x_max,
                y_mm,
            )

            cv2.line(
                image,
                p1,
                p2,
                (20, 20, 20),
                3,
                cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # 실시간 현재 위치 렌더링
    # --------------------------------------------------------

    def render(
        self,
        position_world_mm: Optional[
            np.ndarray
        ] = None,
        yaw_deg: Optional[
            float
        ] = None,
        spread_mm: Optional[
            float
        ] = None,
        used_marker_names: Optional[
            Sequence[str]
        ] = None,
        z_mm: Optional[
            float
        ] = None,
    ) -> np.ndarray:

        image = self._base.copy()

        if position_world_mm is None:
            cv2.putText(
                image,
                "CURRENT POSITION: waiting for registered ArUco...",
                (
                    self.margin,
                    25,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 180),
                2,
                cv2.LINE_AA,
            )

            return image

        position = np.asarray(
            position_world_mm,
            dtype=np.float64,
        ).reshape(3)

        x = float(
            position[0]
        )
        y = float(
            position[1]
        )

        if z_mm is None:
            z_mm = float(
                position[2]
            )

        p = self.world_to_pixel(
            x,
            y,
        )

        # 현재 위치 강조용 외곽 원
        cv2.circle(
            image,
            p,
            11,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            image,
            p,
            8,
            (0, 0, 230),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            image,
            p,
            10,
            (0, 0, 120),
            2,
            cv2.LINE_AA,
        )

        # 카메라 XY 방향
        if yaw_deg is not None:
            radians = np.radians(
                float(yaw_deg)
            )

            direction_world = (
                np.array(
                    [
                        np.cos(radians),
                        np.sin(radians),
                    ],
                    dtype=np.float64,
                )
            )

            arrow_length_mm = 1800.0

            end_world = (
                np.array(
                    [x, y],
                    dtype=np.float64,
                )
                +
                direction_world
                *
                arrow_length_mm
            )

            end_p = (
                self.world_to_pixel(
                    end_world[0],
                    end_world[1],
                )
            )

            cv2.arrowedLine(
                image,
                p,
                end_p,
                (0, 0, 230),
                3,
                cv2.LINE_AA,
                tipLength=0.22,
            )

        # 정보 패널
        cv2.rectangle(
            image,
            (
                self.margin,
                5,
            ),
            (
                min(
                    self.width - self.margin,
                    720,
                ),
                77,
            ),
            (255, 255, 255),
            -1,
        )

        cv2.putText(
            image,
            (
                f"CURRENT: "
                f"x={x:.0f}  "
                f"y={y:.0f}  "
                f"z={float(z_mm):.0f} mm"
            ),
            (
                self.margin + 8,
                27,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 190),
            2,
            cv2.LINE_AA,
        )

        yaw_text = (
            "n/a"
            if yaw_deg is None
            else f"{float(yaw_deg):+.1f} deg"
        )

        spread_text = (
            "n/a"
            if spread_mm is None
            else f"{float(spread_mm):.0f} mm"
        )

        cv2.putText(
            image,
            (
                f"yaw={yaw_text}  "
                f"spread={spread_text}"
            ),
            (
                self.margin + 8,
                51,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )

        if used_marker_names:
            marker_text = (
                "used="
                +
                ",".join(
                    used_marker_names
                )
            )

            cv2.putText(
                image,
                marker_text,
                (
                    self.margin + 8,
                    70,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )

        # 위치가 현재 지도 범위를 벗어났을 때 경고
        if (
            x < MAP_X_MIN
            or x > MAP_X_MAX
            or y < MAP_Y_MIN
            or y > MAP_Y_MAX
        ):
            cv2.putText(
                image,
                "WARNING: calculated position is outside current map range",
                (
                    self.margin,
                    self.height - 28,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 220),
                2,
                cv2.LINE_AA,
            )

        return image
