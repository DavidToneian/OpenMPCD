/**
 * @file
 * Defines exception classes.
 */

#ifndef OPENMPCD_EXCEPTIONS_HPP
#define OPENMPCD_EXCEPTIONS_HPP

#include <cstdio>
#include <string>

/**
 * Throws the given `ExceptionType`, passing the given `message` along with
 * file and line number information to the exception's argument as an instance
 * of `std::string`.
 *
 * @param[in] ExceptionType
 *            The type to throw.
 * @param[in] message
 *            The message to append to file and line number information.
 */
#define OPENMPCD_THROW(ExceptionType, message) do{const char* const _file=__FILE__; const int _line=__LINE__; \
	char _lineString[10]; \
	sprintf(_lineString, "%d", _line); \
	std::string t(message); \
	t+="\r\n\r\nFile: "; \
	t+=_file; t+="\r\nLine: "; \
	t+=_lineString; \
	throw(ExceptionType(t)); \
	} while(0)

namespace OpenMPCD
{
	/**
	 * The base exception class for OpenMPCD.
	 */
	class Exception
	{
		public:
			/**
			 * The constructor.
			 * @param[in] msg The exception message.
			 */
			Exception(const std::string& msg) : message(msg)
			{
			}

		public:
			/**
			 * Returns the exception message.
			 */
			const std::string& getMessage() const
			{
				return message;
			}

		private:
			std::string message; ///< The exception message.
	};

	/**
	 * Represents an exception that is due to an assertion violation.
	 */
	class AssertionException : public Exception
	{
		public:
			/**
			 * The constructor.
			 * @param[in] assertion The assertion that has been violated.
			 */
			AssertionException(const std::string& assertion)
				: Exception("Assertion violated:\n" + assertion)
			{
			}
	};

	/**
	 * Represents an invalid configuration.
	 */
	class InvalidConfigurationException : public Exception
	{
		public:
			/**
			 * The constructor.
			 * @param[in] setting The setting that caused the raising of the exception.
			 */
			InvalidConfigurationException(const std::string& setting)
				: Exception("Inavlid configuration setting: "+setting)
			{
			}
	};

	/**
	 * NULL-pointer exception
	 */
	class NULLPointerException : public Exception
	{
		public:
			/**
			 * The constructor.
			 * @param[in] variable The variable name.
			 */
			NULLPointerException(const std::string& variable)
				: Exception("NULL pointer given: " + variable)
			{
			}
	};

	/**
	 * Exception for out-of-bounds access.
	 */
	class OutOfBoundsException : public Exception
	{
		public:
			/**
			 * The constructor.
			 * @param[in] msg The exception message.
			 */
			OutOfBoundsException(const std::string& msg)
				: Exception("Out of bounds access: "+msg)
			{
			}
	};

	/**
	 * Invalid argument exception.
	 */
	class InvalidArgumentException : public Exception
	{
		public:
			/**
			 * The constructor.
			 * @param[in] msg The exception message.
			 */
			InvalidArgumentException(const std::string& msg)
				: Exception("Invalid argument: "+msg)
			{
			}
	};

	/**
	 * Exception for a forbidden function call.
	 */
	class InvalidCallException : public Exception
	{
		public:
			/**
			 * The constructor.
			 * @param[in] msg The exception message.
			 */
			InvalidCallException(const std::string& msg)
				: Exception(msg)
			{
			}
	};

	/**
	 * Error on IO.
	 */
	class IOException : public Exception
	{
		public:
			/**
			 * The constructor.
			 * @param[in] msg The exception message.
			 */
			IOException(const std::string& msg)
				: Exception(msg)
			{
			}
	};

	/**
	 * Division by zero.
	 */
	class DivisionByZeroException : public Exception
	{
		public:
			/**
			 * The constructor.
			 * @param[in] msg The exception message.
			 */
			DivisionByZeroException(const std::string& msg)
				: Exception(msg)
			{
			}
	};

	/**
	 * Exception for unimplemented functionality.
	 */
	class UnimplementedException : public Exception
	{
		public:
			/**
			 * The constructor.
			 * @param[in] msg The exception message.
			 */
			UnimplementedException(const std::string& msg)
				: Exception(msg)
			{
			}
	};

	/**
	 * Exception for errors in memory management.
	 */
	class MemoryManagementException : public Exception
	{
		public:
			/**
			 * The constructor.
			 * @param[in] msg The exception message.
			 */
			MemoryManagementException(const std::string& msg)
				: Exception(msg)
			{
			}
	};

	/**
	 * Represents an exception that signals a malformed file.
	 */
	class MalformedFileException : public Exception
	{
		public:
			/**
			 * The constructor.
			 * @param[in] msg The exception message.
			 */
			MalformedFileException(const std::string& msg)
				: Exception(msg)
			{
			}
	};
}

#endif
