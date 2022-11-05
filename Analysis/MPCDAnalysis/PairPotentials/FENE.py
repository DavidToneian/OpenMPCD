class FENE:
	r"""
	Models the FENE potential, which is given by
	\f[ V_{\textrm{FENE}}(r) = - 0.5 K R^2 \log (1 - (\frac{r - l_0}{R})^2) \f]
	"""

	def __init__(self, K, l_0, R):
		r"""
		The constructor.

		@throw TypeError
		       Throws if `K`, `l_0`, or `R` are neither `int` nor `float`.
		@throw ValueError
		       Throws if `R` is `0` or negative.

		@param[in] K
		           The \f$ K \f$ potential parameter.
		@param[in] l_0
		           The \f$ l_0 \f$ potential parameter.
		@param[in] R
		           The \f$ R \f$ potential parameter, which must be positive.
		"""

		for var in [K, l_0, R]:
			if not isinstance(var, (int, float)):
				raise TypeError()

		if R <= 0:
			raise ValueError()


		self._K = float(K)
		self._l_0 = float(l_0)
		self._R = float(R)


	def getK(self):
		r"""
		Returns the \f$ K \f$ potential parameter as a `float`.
		"""

		return self._K


	def get_l_0(self):
		r"""
		Returns the \f$ l_0 \f$ potential parameter as a `float`.
		"""

		return self._l_0


	def getR(self):
		r"""
		Returns the \f$ R \f$ potential parameter as a `float`.
		"""

		return self._R


	def getPotential(self, r):
		r"""
		Returns the potential for an input value of \f$ r \f$.

		@throw TypeError
		       Throws if `r` is neither `int` nor `float` or `Vector3DReal`.
		@throw ValueError
		       Throws if `r` is negative.
		@throw ValueError
		       Throws if `r` is such that, in combination with the used
		       potential parameters, the result is undefined.

		@param[in] r
		           The input value. It may be either an `int` or `float`, in
		           which case it must be non-negative. Alternatively, it may be
		           of type `Vector3DReal`, which is then euqivalent to calling
		           `getPotential(r.getLength())` instead.
		"""

		from MPCDAnalysis.Vector3DReal import Vector3DReal
		if isinstance(r, Vector3DReal):
			r = r.getLength()
		else:
			if not isinstance(r, (int, float)):
				raise TypeError()
			if r < 0:
				raise ValueError()

		R = self.getR()
		prefactor = -0.5 * self.getK() * R * R
		frac = (r - self.get_l_0()) / R

		frac2 = frac * frac

		if frac2 >= 1:
			raise ValueError()

		import math
		return prefactor * math.log(1 - frac2)
