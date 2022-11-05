class WeeksChandlerAndersen_DistanceOffset:
	r"""
	A generalization of the Weeks-Chandler-Andersen (WCA) potential.

	The Weeks-Chandler-Andersen potential has been introduced by Weeks,
	Chandler, and Andersen, in J. Chem. Phys. 54, 5237 (1971).
	DOI: 10.1063/1.1674820

	This generalization introduces an offset \f$ D \f$ of the particle distance
	\f$ r \f$. With \f$ \epsilon \f$ and \f$ sigma \f$ being parameters, the
	interaction potential is given by
	\f[
		4 * \epsilon *
		\left(
			\left( \frac{ \sigma }{ r - D } \right)^{12}
			-
			\left( \frac{ \sigma }{ r - D } \right)^{6}
			+
			\frac{ 1 }{ 4 }
		\right)
		*
		\theta \left( 2^{1/6} \sigma - r + D \right)
	\f]
	with \f$ \theta \left( x \right) \f$ being the Heaviside step function,
	which is \f$ 1 \f$ if \f$ x > 0 \f$, and \f$ 0 \f$ otherwise.
	"""

	def __init__(self, epsilon, sigma, d):
		r"""
		The constructor.

		@throw TypeError
		       Throws if `epsilon`, `sigma`, or `d` are neither `int` nor
		       `float`.
		@throw ValueError
		       Throws if `epsilon`, `sigma`, or `d` are negative.

		@param[in] epsilon
		           The \f$ \epsilon \f$ potential parameter, which must be
		           non-negative.
		@param[in] sigma
		           The \f$ \sigma \f$ potential parameter, which must be
		           non-negative.
		@param[in] d
		           The \f$ D \f$ potential parameter, which must be
		           non-negative.
		"""

		for var in [epsilon, sigma, d]:
			if not isinstance(var, (int, float)):
				raise TypeError()
			if var < 0:
				raise ValueError()

		self._epsilon = float(epsilon)
		self._sigma = float(sigma)
		self._d = float(d)


	def getEpsilon(self):
		r"""
		Returns the \f$ \epsilon \f$ potential parameter as a `float`.
		"""

		return self._epsilon


	def getSigma(self):
		r"""
		Returns the \f$ \sigma \f$ potential parameter as a `float`.
		"""

		return self._sigma


	def getD(self):
		r"""
		Returns the \f$ D \f$ potential parameter as a `float`.
		"""

		return self._d


	def getPotential(self, r):
		r"""
		Returns the potential for an input value of \f$ r \f$.

		@throw TypeError
		       Throws if `r` is neither `int` nor `float` or `Vector3DReal`.
		@throw ValueError
		       Throws if `r` is negative.
		@throw ValueError
		       Throws if `r` is smaller than or equal to \f$ D \f$.

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


		d = self.getD()

		if r <= d:
			raise ValueError()

		sigma = self.getSigma()
		sigma2 = sigma * sigma

		cutoff = 2 ** (1.0 / 6.0) * sigma

		if r - d >= cutoff:
			return 0.0

		denominator2 = (r - d) * (r - d)
		frac2 = sigma2 / denominator2
		frac6 = frac2 * frac2 * frac2
		frac12 = frac6 * frac6

		return 4 * self.getEpsilon() * (frac12 - frac6 + 1.0 / 4)
