from argus.data.visdrone import (
    VISDRONE_CLASSES,
    _convert_annotation,
    names_dict,
)


def test_class_count():
    assert len(VISDRONE_CLASSES) == 10
    assert names_dict()[0] == "pedestrian"
    assert names_dict()[3] == "car"


def test_convert_annotation_remaps_and_normalizes(tmp_path):
    # VisDrone line: left,top,w,h,score,category,trunc,occ
    # category 4 ('car') -> training id 3
    ann = tmp_path / "0001.txt"
    ann.write_text("100,200,40,60,1,4,0,0\n")
    lines = _convert_annotation(ann, img_w=1000, img_h=1000)
    assert len(lines) == 1
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "3"
    assert abs(float(cx) - 0.12) < 1e-6  # (100 + 20) / 1000
    assert abs(float(cy) - 0.23) < 1e-6  # (200 + 30) / 1000
    assert abs(float(w) - 0.04) < 1e-6
    assert abs(float(h) - 0.06) < 1e-6


def test_convert_annotation_skips_ignored_and_others(tmp_path):
    ann = tmp_path / "0002.txt"
    # category 0 (ignored) and 11 (others) must be dropped
    ann.write_text("0,0,10,10,1,0,0,0\n10,10,20,20,1,11,0,0\n")
    lines = _convert_annotation(ann, 100, 100)
    assert lines == []


def test_convert_annotation_skips_degenerate_boxes(tmp_path):
    ann = tmp_path / "0003.txt"
    ann.write_text("10,10,0,0,1,4,0,0\n")
    lines = _convert_annotation(ann, 100, 100)
    assert lines == []
