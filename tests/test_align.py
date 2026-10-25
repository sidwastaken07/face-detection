import numpy as np
from PIL import Image

from fpgaface.data.align import ARCFACE_TEMPLATE, align_face, similarity_transform


def test_similarity_recovers_known_transform():
    theta, s, t = 0.3, 1.7, np.array([5.0, -3.0])
    r = s * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    dst = ARCFACE_TEMPLATE @ r.T + t
    m = similarity_transform(ARCFACE_TEMPLATE, dst)
    assert np.allclose(m[:, :2], r) and np.allclose(m[:, 2], t)


def test_align_face_moves_landmark_to_template():
    # A bright dot at the "left eye" position of a scaled/shifted face must land on the template.
    lm = ARCFACE_TEMPLATE * 2 + [20, 30]
    img = np.zeros((300, 300, 3), np.uint8)
    x, y = lm[0].round().astype(int)
    img[y - 2 : y + 3, x - 2 : x + 3] = 255
    out = np.asarray(align_face(Image.fromarray(img), lm, 112))[..., 0]
    yy, xx = np.unravel_index(np.argmax(out), out.shape)
    assert abs(xx - ARCFACE_TEMPLATE[0, 0]) <= 1.5 and abs(yy - ARCFACE_TEMPLATE[0, 1]) <= 1.5
