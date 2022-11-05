/**
 * @file
 * Declares functions for the computeTransverseVelocityCorrelationFunction program.
 */

#ifndef COMPUTETRANSVERSEVELOCITYCORRELATIONFUNCTION_FUNCTIONS_HPP
#define COMPUTETRANSVERSEVELOCITYCORRELATIONFUNCTION_FUNCTIONS_HPP

#include <OpenMPCD/Vector3D.hpp>

#include <map>
#include <string>
#include <vector>

/**
 * Processes the given run directories and writes the corresponding plotfiles.
 *
 * @param[in] minimumTime
 *            All data that are taken at simulation times smaller than this
 *            argument are ignored for the analysis. Useful to allow for a
 *            "warm-up" period.
 *
 * @param[in] maxCorrelationTime The maximum correlation time to compute results for.
 * @param[in] rundirs            A list of run directories to process.
 * @param[in] pathBasename       A template string for the plotfile paths,
 *                               to which the integer components of the k-vector
 *                               are appended.
 * @param[in]  metadataFilename  If not empty, path to write metadata to.
 * @param[in]  metadataTableFilename
 *                               If not empty, path to write a subset of the
 *                               metadata to, in tabular form.
 * @param[out] ostream           Pointer to an output stream to write status
 *                               messages to, or `nullptr` to suppress output.
 */
void writePlotfiles(
	const double minimumTime,
	const double maxCorrelationTime, const std::vector<std::string>& rundirs,
	const std::string& pathBasename, const std::string& metadataFilename,
	const std::string& metadataTableFilename,
	std::ostream* const ostream);

/**
 * Returns the list of available k-vectors, along with the associated file paths.
 *
 * The k-vectors' components are understood to be multiples of 2*pi/L, where L is the simulation
 * box size along the corresponding cartesian direction.
 *
 * @param[in] rundirs The run directories to search.
 * @throw OpenMPCD::IOException Throws if the first line of the file does not describe the k-vector used.
 */
const std::map<OpenMPCD::Vector3D<double>, std::vector<std::string>>
	getWaveVectorsAndPaths(const std::vector<std::string>& rundirs);

#endif
