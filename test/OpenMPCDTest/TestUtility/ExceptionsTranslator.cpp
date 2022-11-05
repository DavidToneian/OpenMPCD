/**
 * @file
 * Adds a Catch framework translator for `OpenMPCD::Exception`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/Exceptions.hpp>

CATCH_TRANSLATE_EXCEPTION(const OpenMPCD::Exception& e)
{
	return e.getMessage();
}
