class MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles:
	r"""
	Interactions between two constant and identical magnetic dipoles.

	The general magnetic dipole-dipole interaction potential is given by
	\f[
		- \frac{ \mu_0 }{ 4 \pi r^3 }
		\left(
			3
			\left(\vec{m_1} \cdot \hat{r} \right)
			\left(\vec{m_2} \cdot \hat{r} \right)
			-
			\vec{m_1} \cdot \vec{m_2}
		\right)
	\f]
	where \f$ \mu_0 \f$ is the vacuum permeability, \f$ \hat{r} \f$ and
	\f$ r \f$ are, respectively, the unit vector and length of the vector
	\f$ \vec{r} \f$ that points from one dipole's position to the other's,
	\f$ \vec{m_1} \f$ and \f$ \vec{m_2} \f$ are the magnetic dipole moments, and
	\f$ \cdot \f$ denotes the inner product.

	In the special case treated in this class, the magnetic dipole moments are
	assumed to be constant throughout time in size and orientation. Therefore,
	with \f$ m \f$ being the magnitude of the individual dipole moments and with
	\f$ \hat{m} \f$ being the unit vector of the individual dipole moments, the
	interaction potential is given by
	\f[
		- \frac{ \mu_0 m^2 }{ 4 \pi r^3 }
		\left( 3 \left(\hat{m} \cdot \hat{r} \right)^2 - 1 \right)
	\f]
	"""

	def __init__(self, prefactor, orientation):
		r"""
		The constructor.

		@throw TypeError
		       Throws if `prefactor` is neither `int` nor `float`.
		@throw TypeError
		       Throws if `orientation` is not an instance of `Vector3DReal`.
		@throw ValueError
		       Throws if `prefactor` is negative.
		@throw ValueError
		       Throws if `orientation` is not a unit vector.

		@param[in] prefactor
		           The term \f$ \frac{\mu_0 m^2}{4 \pi} \f$, which must be
		           non-negative.
		@param[in] orientation
		           The orientation unit vector \f$ \hat{m} \f$ of the dipole
		           moments.
		"""

		from MPCDAnalysis.Vector3DReal import Vector3DReal

		if not isinstance(prefactor, (int, float)):
			raise TypeError()
		if not isinstance(orientation, Vector3DReal):
			raise TypeError()

		if prefactor < 0:
			raise ValueError()
		if orientation.getLengthSquared() != 1.0:
			raise ValueError()

		import copy
		self._prefactor = float(prefactor)
		self._orientation = copy.deepcopy(orientation)


	def getPrefactor(self):
		r"""
		Returns the term \f$ \frac{\mu_0 m^2}{4 \pi} \f$ as a `float`.
		"""

		return self._prefactor


	def getOrientation(self):
		r"""
		Returns the orientation unit vector \f$ \hat{m} \f$ of the dipole
		moments as an instance of `Vector3DReal`.
		"""

		return self._orientation


	def getPotential(self, r):
		r"""
		Returns the potential for an input value of \f$ r \f$.

		@throw TypeError
		       Throws if `r` is neither `int` nor `float` or `Vector3DReal`.
		@throw ValueError
		       Throws if `r` is the zero vector.

		@param[in] r
		           The input value, which must may be a non-zero vector of type
		           `Vector3DReal`.
		"""

		from MPCDAnalysis.Vector3DReal import Vector3DReal

		if not isinstance(r, Vector3DReal):
			raise TypeError

		r2 = r.getLengthSquared()

		if r2 == 0:
			raise ValueError()

		import math

		r_m2 = 1.0 / r2
		r_m3 = r_m2 / math.sqrt(r2)
		dot = r.dot(self.getOrientation())

		return -self.getPrefactor() * r_m3 * (3 * dot * dot * r_m2 - 1)
