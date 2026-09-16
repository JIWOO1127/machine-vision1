"""
aruco_world_map.py

2026-09-16 현장 재확인 값을 반영한 ArUco 월드 맵.

좌표계
------
+X : 평면도 오른쪽
-X : 평면도 왼쪽
+Y : 평면도 위쪽
-Y : 평면도 아래쪽
+Z : 바닥 -> 천장
-Z : 천장 -> 바닥

마커 방향
---------
face_world:
    마커의 앞면(카메라가 마커를 볼 수 있는 쪽)이 향하는 월드 방향.
    즉 marker local +Z.

top_world:
    인쇄된 ArUco 그림의 TOP이 향하는 월드 방향.
    즉 marker local +Y.

right_world:
    marker local +X. face_world/top_world로 자동 계산한다.

중요:
    마커 회전은 installation 문자열이나 벽 이름에서 추론하지 않는다.
    현장에서 확인한 face_world/top_world만 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


WORLD_AXIS_VECTORS: Dict[str, np.ndarray] = {
    "+X": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "-X": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "+Y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "-Y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "+Z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "-Z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
}


@dataclass(frozen=True)
class ArucoMarker:
    point_name: str
    name: str
    description: str
    marker_id: int
    size_mm: float

    x: float
    y: float
    z: float

    face_world: str
    top_world: str

    @property
    def position_mm(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.z],
            dtype=np.float64,
        )

    @property
    def plane(self) -> str:
        """
        face 방향에서 설치 평면을 자동 결정.
        ±Z normal -> XY
        ±Y normal -> XZ
        ±X normal -> YZ
        """
        axis = self.face_world[-1].upper()

        if axis == "Z":
            return "XY"
        if axis == "Y":
            return "XZ"
        if axis == "X":
            return "YZ"

        raise ValueError(
            f"{self.point_name}: 잘못된 face_world={self.face_world}"
        )

    @property
    def right_world(self) -> str:
        """
        marker local coordinate:
            +X = 그림의 RIGHT
            +Y = 그림의 TOP
            +Z = 마커 FRONT/FACE

        오른손 좌표계이므로:
            right = top × face
        """
        top = WORLD_AXIS_VECTORS[self.top_world]
        face = WORLD_AXIS_VECTORS[self.face_world]

        right = np.cross(top, face)

        for name, vector in WORLD_AXIS_VECTORS.items():
            if np.allclose(right, vector):
                return name

        raise ValueError(
            f"{self.point_name}: TOP과 FACE가 "
            f"직교하지 않습니다. "
            f"top={self.top_world}, face={self.face_world}"
        )

    def rotation_marker_to_world(self) -> np.ndarray:
        """
        marker local -> world rotation matrix.

        열(column):
            0: marker +X (RIGHT) in world
            1: marker +Y (TOP)   in world
            2: marker +Z (FACE)  in world
        """
        top = WORLD_AXIS_VECTORS[self.top_world]
        face = WORLD_AXIS_VECTORS[self.face_world]
        right = np.cross(top, face)

        R_wm = np.column_stack(
            (right, top, face)
        ).astype(np.float64)

        # 데이터 오류를 즉시 잡기 위한 검증.
        if not np.allclose(
            R_wm.T @ R_wm,
            np.eye(3),
            atol=1e-8,
        ):
            raise ValueError(
                f"{self.point_name}: 회전축이 직교하지 않습니다."
            )

        determinant = float(
            np.linalg.det(R_wm)
        )

        if not np.isclose(
            determinant,
            1.0,
            atol=1e-8,
        ):
            raise ValueError(
                f"{self.point_name}: "
                f"오른손 회전행렬이 아닙니다. det={determinant}"
            )

        return R_wm


# ============================================================
# 최신 현장 확인값
# ============================================================
#
# 위치는 지금까지의 최신 좌표를 유지.
# 방향/ID/크기는 2026-09-16 현장 재확인 값을 우선 적용.
#
ARUCO_MARKERS: Tuple[ArucoMarker, ...] = (

    ArucoMarker(
        point_name="P01",
        name="서측 복도 바닥 마커",
        description="현재 지도에서 가장 서쪽에 있는 바닥 ArUco 마커.",
        marker_id=25,
        size_mm=134.0,
        x=-3760.0,
        y=1240.0,
        z=0.0,
        face_world="+Z",
        top_world="-Y",
    ),

    ArucoMarker(
        point_name="P02",
        name="서측 y=1962 벽 마커",
        description="서측 구간 y=1962 경계에 설치된 작은 벽 마커.",
        marker_id=9,
        size_mm=34.0,
        x=450.0,
        y=1962.0,
        z=800.0,
        face_world="-Y",
        top_world="-Z",
    ),

    ArucoMarker(
        point_name="P03",
        name="4번강의실 인근 y=0 벽 마커",
        description="4번강의실 출입구 인근 y=0 벽면의 ArUco 마커.",
        marker_id=13,
        size_mm=134.0,
        x=1280.0,
        y=0.0,
        z=1800.0,
        face_world="+Y",
        top_world="+Z",
    ),

    ArucoMarker(
        point_name="P04",
        name="3번강의실 서측 바닥 마커",
        description="3번강의실 쪽으로 이동하는 복도 바닥의 ArUco 마커.",
        marker_id=14,
        size_mm=134.0,
        x=8930.0,
        y=740.0,
        z=0.0,
        face_world="+Z",
        top_world="-Y",
    ),

    ArucoMarker(
        point_name="P05",
        name="3번강의실 인근 y=0 벽 마커",
        description="3번강의실 출입구 인근 y=0 벽면의 ArUco 마커.",
        marker_id=15,
        size_mm=134.0,
        x=10980.0,
        y=0.0,
        z=1800.0,
        face_world="+Y",
        top_world="-X",
    ),

    ArucoMarker(
        point_name="P06",
        name="2번강의실 인근 y=1962 벽 마커",
        description=(
            "현장 재측정으로 위치를 (19195,1962,820)으로 수정했고, "
            "크기는 34 mm로 재확인한 벽 마커."
        ),
        marker_id=17,
        size_mm=34.0,
        x=19195.0,
        y=1962.0,
        z=820.0,
        face_world="-Y",
        top_world="-Z",
    ),

    ArucoMarker(
        point_name="P07",
        name="2번강의실 인근 y=0 벽 마커",
        description="2번강의실 출입구 인근 y=0 벽면의 ArUco 마커.",
        marker_id=4,
        size_mm=134.0,
        x=20440.0,
        y=0.0,
        z=180.0,
        face_world="+Y",
        top_world="+X",
    ),

    ArucoMarker(
        point_name="P08",
        name="2번강의실 동측 바닥 마커",
        description="2번강의실을 지나 동쪽 복도 바닥에 설치된 ArUco 마커.",
        marker_id=2,
        size_mm=134.0,
        x=21080.0,
        y=750.0,
        z=0.0,
        face_world="+Z",
        top_world="+Y",
    ),

    ArucoMarker(
        point_name="P09",
        name="동측 중앙 복도 바닥 마커",
        description="2번강의실과 쪽문 사이 구간 바닥에 설치된 ArUco 마커.",
        marker_id=5,
        size_mm=134.0,
        x=20610.0,
        y=3570.0,
        z=0.0,
        face_world="+Z",
        top_world="-X",
    ),

    ArucoMarker(
        point_name="P10",
        name="동쪽 끝 x=27660 벽 마커",
        description=(
            "지도 동쪽 끝의 x=27660 세로벽 마커. "
            "현장 재확인으로 ArUco ID를 10으로 수정."
        ),
        marker_id=10,
        size_mm=134.0,
        x=27660.0,
        y=5862.0,
        z=163.0,
        face_world="-X",
        top_world="+Y",
    ),

    ArucoMarker(
        point_name="P11",
        name="x=19475 세로벽 마커",
        description=(
            "테이블/정수기/벽 블록 쪽 x=19475 세로 경계의 마커. "
            "현장 재확인으로 ID 12, 크기 134 mm로 수정. "
            "중간 장애물로 인한 가려짐 개선을 위해 +Y 방향으로 260 mm 이동."
        ),
        marker_id=12,
        size_mm=134.0,
        x=19475.0,
        y=6012.0,
        z=1250.0,
        face_world="+X",
        top_world="-Y",
    ),

    ArucoMarker(
        point_name="P12",
        name="쪽문 벽 마커",
        description="y=6792 쪽문 부근에 설치된 작은 벽 마커.",
        marker_id=8,
        size_mm=34.0,
        x=20235.0,
        y=6792.0,
        z=1330.0,
        face_world="-Y",
        top_world="+Z",
    ),
)


ARUCO_MARKERS_BY_ID: Dict[int, ArucoMarker] = {
    marker.marker_id: marker
    for marker in ARUCO_MARKERS
}

ARUCO_MARKERS_BY_POINT: Dict[str, ArucoMarker] = {
    marker.point_name: marker
    for marker in ARUCO_MARKERS
}


def get_marker_by_id(marker_id: int) -> Optional[ArucoMarker]:
    return ARUCO_MARKERS_BY_ID.get(
        int(marker_id)
    )


def get_marker(point_name: str) -> Optional[ArucoMarker]:
    return ARUCO_MARKERS_BY_POINT.get(
        point_name.upper()
    )


def validate_marker_database() -> None:
    ids = [
        marker.marker_id
        for marker in ARUCO_MARKERS
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "중복된 ArUco ID가 있습니다."
        )

    names = [
        marker.point_name
        for marker in ARUCO_MARKERS
    ]

    if len(names) != len(set(names)):
        raise ValueError(
            "중복된 P번호가 있습니다."
        )

    for marker in ARUCO_MARKERS:
        if marker.face_world not in WORLD_AXIS_VECTORS:
            raise ValueError(
                f"{marker.point_name}: face_world 오류"
            )

        if marker.top_world not in WORLD_AXIS_VECTORS:
            raise ValueError(
                f"{marker.point_name}: top_world 오류"
            )

        face = WORLD_AXIS_VECTORS[
            marker.face_world
        ]

        top = WORLD_AXIS_VECTORS[
            marker.top_world
        ]

        if not np.isclose(
            float(np.dot(face, top)),
            0.0,
        ):
            raise ValueError(
                f"{marker.point_name}: "
                f"FACE와 TOP이 직교하지 않습니다."
            )

        marker.rotation_marker_to_world()


def print_marker_table() -> None:
    header = (
        "P    ID   SIZE   POSITION(mm)                 "
        "FACE  TOP   RIGHT PLANE  NAME"
    )

    print(header)
    print("-" * len(header))

    for m in ARUCO_MARKERS:
        print(
            f"{m.point_name:<4} "
            f"{m.marker_id:<4} "
            f"{m.size_mm:>5.0f}  "
            f"({m.x:>7.0f},{m.y:>6.0f},{m.z:>5.0f})   "
            f"{m.face_world:<5} "
            f"{m.top_world:<5} "
            f"{m.right_world:<5} "
            f"{m.plane:<5}  "
            f"{m.name}"
        )


validate_marker_database()


if __name__ == "__main__":
    print_marker_table()
