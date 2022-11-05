from MPCDAnalysis.Vector3DReal import Vector3DReal

def test___init___getX_getY_getZ():
	import pytest

	v = Vector3DReal(-1, 2, 3.4)
	assert v.getX() == -1
	assert v.getY() == 2
	assert v.getZ() == 3.4

	v2 = Vector3DReal(v)
	assert v2.getX() == -1
	assert v2.getY() == 2
	assert v2.getZ() == 3.4

	with pytest.raises(TypeError):
		Vector3DReal(v, 0)

	v = Vector3DReal([-10, 20.5, 30])
	assert v.getX() == -10
	assert v.getY() == 20.5
	assert v.getZ() == 30

	with pytest.raises(TypeError):
		Vector3DReal([0, 0, 0], 0)

	with pytest.raises(TypeError):
		Vector3DReal([0, 0])

	with pytest.raises(TypeError):
		Vector3DReal([0, 0, 0, 0])


def test_dot():
	v = Vector3DReal(1, 2, 3)
	v2 = Vector3DReal(-1, 1, 2)
	d = v.dot(v2)
	assert d == v2.dot(v)
	assert \
		d == v.getX() * v2.getX() + v.getY() * v2.getY() + v.getZ() * v2.getZ()


def test_cross():
	v = Vector3DReal(1, 2, 3)
	v2 = Vector3DReal(-1, 1, 2)

	c = v.cross(v2)
	assert c == -v2.cross(v)
	assert c.getX() == 1
	assert c.getY() == -5
	assert c.getZ() == 3


def test_getLengthSquared():
	v = Vector3DReal(1, 2, 3)
	n = Vector3DReal(0, 0, 0)

	assert v.getLengthSquared() == v.dot(v)
	assert n.getLengthSquared() == 0


def test_getLength():
	import math

	v = Vector3DReal(1, 2, 3)
	n = Vector3DReal(0, 0, 0)

	assert v.getLength() == math.sqrt(v.getLengthSquared())
	assert n.getLength() == 0


def test_getNormalized():
	import pytest

	v = Vector3DReal(1, 2, 3)
	n = Vector3DReal(0, 0, 0)

	assert v.getNormalized() == v / v.getLength()
	with pytest.raises(ValueError):
		n.getNormalized()


def test_normalize():
	import pytest

	v = Vector3DReal(1, 2, 3)
	n = Vector3DReal(0, 0, 0)

	v.normalize()
	assert v.getLengthSquared() == 1

	with pytest.raises(ValueError):
		n.normalize()


def test_getProjectionOnto():
	import pytest

	v = Vector3DReal(1, 2.5, 3)
	v2 = Vector3DReal(-1, 1, 1.5)
	n = Vector3DReal(0, 0, 0)

	p = v.getProjectionOnto(v2)
	assert p == v2.getNormalized() * (v.dot(v2.getNormalized()))
	assert n.getProjectionOnto(v2).getLength() == 0

	with pytest.raises(ValueError):
		v.getProjectionOnto(n)


def test_getPerpendicularTo():
	import pytest

	v = Vector3DReal(1, 2.5, 3)
	v2 = Vector3DReal(-1, 1, 1.5)
	n = Vector3DReal(0, 0, 0)

	perp = v.getPerpendicularTo(v2)
	p = v.getProjectionOnto(v2)
	assert perp == v - p
	assert n.getPerpendicularTo(v2).getLength() == 0

	with pytest.raises(ValueError):
		v.getPerpendicularTo(n)


def test_getRotatedAroundNormalizedAxis():
	x = Vector3DReal(1, 0, 0)
	z = Vector3DReal(0, 0, 1)

	import numpy
	import math

	r = x.getRotatedAroundNormalizedAxis(z, math.pi)
	assert numpy.isclose(r.getX(), -1)
	assert numpy.isclose(r.getY(), 0)
	assert numpy.isclose(r.getZ(), 0)

	r = x.getRotatedAroundNormalizedAxis(z, math.pi / 2)
	assert numpy.isclose(r.getX(), 0)
	assert numpy.isclose(r.getY(), 1)
	assert numpy.isclose(r.getZ(), 0)


def test_rotateAroundNormalizedAxis():
	import math

	x = Vector3DReal(1, 0, 0)
	z = Vector3DReal(0, 0, 1)
	r = x.getRotatedAroundNormalizedAxis(z, math.pi / 2)

	x.rotateAroundNormalizedAxis(z, math.pi / 2)
	assert x == r


def test_isClose():
	v1 = Vector3DReal(-1, 2.5, 3.0)
	v2 = Vector3DReal(-1.2, 2.5, 3.0)

	assert v1.isClose(v1)
	assert v2.isClose(v2)
	assert not v1.isClose(v2)
	assert not v2.isClose(v1)

	v1 = Vector3DReal(1, 1e10, 1e-7)
	v2 = Vector3DReal(1, 1.00001e10, 1e-8)
	assert not v1.isClose(v2)
	assert not v2.isClose(v1)

	v1 = Vector3DReal(1, 1e10, 1e-8)
	v2 = Vector3DReal(1, 1.00001e10, 1e-9)
	assert v1.isClose(v2)
	assert v2.isClose(v1)

	v1 = Vector3DReal(1, 1e10, 1e-8)
	v2 = Vector3DReal(1, 1.0001e10, 1e-9)
	assert not v1.isClose(v2)
	assert not v2.isClose(v1)

	assert v1.isClose(v2, relativeTolerance = 1e-4)
	assert v2.isClose(v1, relativeTolerance = 1e-4)

	assert v1.isClose(v2, absoluteTolerance = 1e6)
	assert v2.isClose(v1, absoluteTolerance = 1e6)


def test___eq__():
	v = Vector3DReal(-1, 2.5, 3.0)

	assert v == v
	assert v == Vector3DReal(-1.0, 2.5, 3)
	assert v != Vector3DReal(0, 0, 0)

	import pytest
	with pytest.raises(TypeError):
		v == (-1, 2.5, 3.0)


def test___ne__():
	v = Vector3DReal(-1, 2.5, 3.0)

	assert not v != v
	assert not v != Vector3DReal(-1.0, 2.5, 3)
	assert v != Vector3DReal(0, 0, 0)


def test___add__():
	v = Vector3DReal(-1, 2.5, 3.0)
	v2 = Vector3DReal(5, 6.7, 8)
	assert v + v2 == Vector3DReal(4, 9.2, 11)

	import pytest
	with pytest.raises(TypeError):
		v + (-1, 2.5, 3.0)


def test___sub__():
	v = Vector3DReal(-1, 2.5, 3.0)
	v2 = Vector3DReal(5, 6.7, 8)

	assert v - v2 == Vector3DReal(-6, -4.2, -5)

	import pytest
	with pytest.raises(TypeError):
		v - (-1, 2.5, 3.0)


def test___mul__():
	v = Vector3DReal(-1, 2.5, 3.0)

	assert v * 3 == Vector3DReal(-3, 7.5, 9.0)

	import pytest
	with pytest.raises(TypeError):
		v * v


def test___div_____truediv__():
	v = Vector3DReal(-1, 2.5, 3.0)

	assert v / 2 == Vector3DReal(-0.5, 1.25, 1.5)
	assert v / 2.0 == Vector3DReal(-0.5, 1.25, 1.5)
	assert Vector3DReal(1, 2, 3) / 2 == Vector3DReal(0.5, 1.0, 1.5)
	assert Vector3DReal(1, 2, 3) / 2.0 == Vector3DReal(0.5, 1.0, 1.5)
	assert Vector3DReal(1.0, 2, 3) / 2 == Vector3DReal(0.5, 1.0, 1.5)
	assert Vector3DReal(1.0, 2, 3) / 2.0 == Vector3DReal(0.5, 1.0, 1.5)
	assert Vector3DReal(1.0, 2.0, 3.0) / 2 == Vector3DReal(0.5, 1.0, 1.5)
	assert Vector3DReal(1.0, 2.0, 3.0) / 2.0 == Vector3DReal(0.5, 1.0, 1.5)

	import pytest
	with pytest.raises(TypeError):
		v / v

	with pytest.raises(ValueError):
		v / 0


def test___neg__():
	v = Vector3DReal(-1, 2.5, 3.0)

	assert -v == Vector3DReal(-v.getX(), -v.getY(), -v.getZ())


def test___getitem__():
	v = Vector3DReal(-1, 2.5, 3.0)

	assert v[0] == v.getX()
	assert v[1] == v.getY()
	assert v[2] == v.getZ()

	import pytest
	with pytest.raises(KeyError):
		v[3]


def test___repr__():
	v = Vector3DReal(-1, 2.5, 3.0)

	assert isinstance(v.__repr__(), str)


def test__assertIsSameType():
	v = Vector3DReal(-1, 2.5, 3.0)

	v._assertIsSameType(v)

	import pytest
	with pytest.raises(TypeError):
		v._assertIsSameType(2)
