"""
check_marker_database.py

실행:
    python check_marker_database.py

최신 마커 ID/크기/좌표/FACE/TOP/RIGHT를 콘솔에서 확인한다.
"""

from aruco_world_map import (
    print_marker_table,
    validate_marker_database,
)

validate_marker_database()
print_marker_table()
print()
print("Marker database validation: OK")
