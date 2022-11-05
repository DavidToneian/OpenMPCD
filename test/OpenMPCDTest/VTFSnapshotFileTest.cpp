/**
 * @file
 * Tests `OpenMPCD::VTFSnapshotFile`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/VTFSnapshotFile.hpp>

#include <OpenMPCD/Exceptions.hpp>

#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>

using OpenMPCD::VTFSnapshotFile;

SCENARIO(
	"`OpenMPCD::VTFSnapshotFile::VTFSnapshotFile`, "
	"`OpenMPCD::VTFSnapshotFile::isInWriteMode`, "
	"`OpenMPCD::VTFSnapshotFile::isInReadMode`",
	"")
{
	static const char* const path = "test/data/VTFSnapshotFile/tmp.vtf";

	WHEN("the constructor is given a good path")
	{
		AND_WHEN("the file exists already and is accessible")
		{
			//make sure file exists and is empty
			boost::filesystem::remove(path);
			(boost::filesystem::ofstream(path));
			REQUIRE(boost::filesystem::is_regular_file(path));

			THEN("the snapshot is opened in read mode")
			{
				VTFSnapshotFile snapshot(path);

				REQUIRE(snapshot.isInReadMode());
				REQUIRE(!snapshot.isInWriteMode());
			}
		}

		AND_WHEN("the file does not exist yet")
		{
			boost::filesystem::remove(path);
			REQUIRE(!boost::filesystem::is_regular_file(path));

			THEN("the file is created, "
			     "and the snapshot is opened in write mode")
			{
				VTFSnapshotFile snapshot(path);

				REQUIRE(!snapshot.isInReadMode());
				REQUIRE(snapshot.isInWriteMode());

				REQUIRE(boost::filesystem::is_regular_file(path));
			}
		}
	}

	WHEN("the constructor is given a path that is inaccessible")
	{
		THEN("`OpenMPCD::IOException` is thrown")
		{
			const std::string inaccessiblePath = "/inaccessible/path";

			REQUIRE_THROWS_AS(
				(VTFSnapshotFile(inaccessiblePath)),
				OpenMPCD::IOException);
		}
	}
}


SCENARIO(
	"`OpenMPCD::VTFSnapshotFile::structureBlockHasBeenProcessed`",
	"")
{
	static const char* const readPath =
		"test/data/VTFSnapshotFile/timestep-blocks-ordered.vtf";
	static const char* const writePath =
		"test/data/VTFSnapshotFile/tmp.vtf";

	WHEN("an instance is constructed in read mode")
	{
		REQUIRE(boost::filesystem::is_regular_file(readPath));
		const VTFSnapshotFile snapshot(readPath);

		THEN("the structure block has been read")
		{
			REQUIRE(snapshot.structureBlockHasBeenProcessed());
		}
	}

	WHEN("an instance is constructed in write mode")
	{
		boost::filesystem::remove(writePath);
		REQUIRE(!boost::filesystem::is_regular_file(writePath));

		VTFSnapshotFile snapshot(writePath);

		THEN("the structure block has not yet been processed")
		{
			REQUIRE_FALSE(snapshot.structureBlockHasBeenProcessed());
		}

		AND_WHEN("stucture information is supplied")
		{
			static const std::size_t atomCount = 3;
			snapshot.setPrimarySimulationVolumeSize(1, 2, 3);
			snapshot.declareAtoms(atomCount);

			THEN("the structure block still has not been processed")
			{
				REQUIRE_FALSE(snapshot.structureBlockHasBeenProcessed());
			}

			AND_WHEN("`writeTimestepBlock` is called")
			{
				const OpenMPCD::FP positions[atomCount * 3] = {0};
				snapshot.writeTimestepBlock(positions);

				THEN("the structure block will have been processed")
				{
					REQUIRE(snapshot.structureBlockHasBeenProcessed());
				}
			}
		}
	}
}


SCENARIO(
	"`OpenMPCD::VTFSnapshotFile::setPrimarySimulationVolumeSize`, "
	"`OpenMPCD::VTFSnapshotFile::primarySimulationVolumeSizeIsSet`, "
	"`OpenMPCD::VTFSnapshotFile::getPrimarySimulationVolumeSize`",
	"")
{
	using OpenMPCD::FP;
	using OpenMPCD::Vector3D;

	GIVEN("a snapshot file in write mode")
	{
		static const char* const path = "test/data/VTFSnapshotFile/tmp.vtf";

		boost::filesystem::remove(path);
		boost::scoped_ptr<VTFSnapshotFile> snapshot(new VTFSnapshotFile(path));

		REQUIRE(snapshot->isInWriteMode());

		THEN("`primarySimulationVolumeSizeIsSet` returns `false`")
		{
			REQUIRE(!snapshot->primarySimulationVolumeSizeIsSet());
		}

		THEN("`getPrimarySimulationVolumeSize` throws")
		{
			REQUIRE_THROWS_AS(
				snapshot->getPrimarySimulationVolumeSize(),
				OpenMPCD::InvalidCallException);
		}

		THEN("`setPrimarySimulationVolumeSize` can be called")
		{
			const Vector3D<FP> expected(-1.23, 0, 4e8);

			snapshot->setPrimarySimulationVolumeSize(
				expected.getX(), expected.getY(), expected.getZ());

			AND_THEN("`primarySimulationVolumeSizeIsSet` is `true`")
			{
				REQUIRE(snapshot->primarySimulationVolumeSizeIsSet());
			}

			AND_THEN("`getPrimarySimulationVolumeSize` returns expected values")
			{
				REQUIRE(snapshot->getPrimarySimulationVolumeSize() == expected);
			}

			AND_THEN(
				"`getPrimarySimulationVolumeSize` on a new instance "
				"returns the correct values")
			{
				snapshot.reset();
				VTFSnapshotFile readsnap(path);

				REQUIRE(readsnap.isInReadMode());
				REQUIRE(readsnap.primarySimulationVolumeSizeIsSet());
				REQUIRE(readsnap.getPrimarySimulationVolumeSize() == expected);
			}
		}
	}

	GIVEN("a snapshot file in read mode")
	{
		const char* const primSimVolPath =
			"test/data/VTFSnapshotFile/primarySimulationVolume.vtf";

		REQUIRE(boost::filesystem::is_regular_file(primSimVolPath));

		VTFSnapshotFile snapshot(primSimVolPath);

		THEN("`primarySimulationVolumeSizeIsSet` returns `true`")
		{
			REQUIRE(snapshot.primarySimulationVolumeSizeIsSet());
		}

		THEN("`setPrimarySimulationVolumeSize` throws")
		{
			REQUIRE_THROWS_AS(
				snapshot.setPrimarySimulationVolumeSize(0, 0, 0),
				OpenMPCD::InvalidCallException);
		}

		THEN("`getPrimarySimulationVolumeSize` returns correct data")
		{
			const Vector3D<FP> expected(-3.1415, 0, 9.87e3);

			REQUIRE(snapshot.getPrimarySimulationVolumeSize() == expected);
		}
	}
}


SCENARIO(
	"`OpenMPCD::VTFSnapshotFile::declareAtoms`, "
	"`OpenMPCD::VTFSnapshotFile::getNumberOfAtoms`, "
	"`OpenMPCD::VTFSnapshotFile::isValidAtomID` "
	"without properties",
	"")
{
	GIVEN("a snapshot file in write mode")
	{
		static const char* const path =
			"test/data/VTFSnapshotFile/tmp-atom-lines.vtf";

		boost::filesystem::remove(path);
		boost::scoped_ptr<VTFSnapshotFile> snapshot(new VTFSnapshotFile(path));

		REQUIRE(snapshot->isInWriteMode());

		THEN("`getNumberOfAtoms` returns `0`")
		{
			REQUIRE(snapshot->getNumberOfAtoms() == 0);
		}

		THEN("`isValidAtomID` returns `false`")
		{
			REQUIRE(!snapshot->isValidAtomID(0));
			REQUIRE(!snapshot->isValidAtomID(1));
			REQUIRE(!snapshot->isValidAtomID(123456789));
		}

		THEN("`declareAtoms` can be called")
		{
			const std::pair<std::size_t, std::size_t> declaredAtoms1 =
				snapshot->declareAtoms(120);
			const std::pair<std::size_t, std::size_t> declaredAtoms2 =
				snapshot->declareAtoms(3);
			const std::pair<std::size_t, std::size_t> declaredAtoms3 =
				snapshot->declareAtoms(0);

			AND_THEN("`declareAtoms` returns the right values")
			{
				REQUIRE(declaredAtoms1.first == 0);
				REQUIRE(declaredAtoms1.second == 120 - 1);
				REQUIRE(declaredAtoms2.first == 120);
				REQUIRE(declaredAtoms2.second == 123 - 1);

				REQUIRE(declaredAtoms3.first == 0);
				REQUIRE(declaredAtoms3.second == 0);
			}

			AND_THEN("`getNumberOfAtoms` returns `123`")
			{
				REQUIRE(snapshot->getNumberOfAtoms() == 123);
			}

			AND_THEN("`isValidAtomID` returns `true` for IDs < 123")
			{
				REQUIRE(snapshot->isValidAtomID(0));
				REQUIRE(snapshot->isValidAtomID(1));
				REQUIRE(snapshot->isValidAtomID(5));
				REQUIRE(snapshot->isValidAtomID(122));
				REQUIRE(!snapshot->isValidAtomID(123));
				REQUIRE(!snapshot->isValidAtomID(1234));
			}

			AND_THEN(
				"`getNumberOfAtoms` on a new instance "
				"returns the correct value, `isValidAtomID` works as expected")
			{
				snapshot.reset();
				VTFSnapshotFile readsnap(path);

				REQUIRE(readsnap.isInReadMode());
				REQUIRE(readsnap.getNumberOfAtoms() == 123);

				REQUIRE(readsnap.isValidAtomID(0));
				REQUIRE(readsnap.isValidAtomID(1));
				REQUIRE(readsnap.isValidAtomID(5));
				REQUIRE(readsnap.isValidAtomID(122));
				REQUIRE(!readsnap.isValidAtomID(123));
				REQUIRE(!readsnap.isValidAtomID(1234));
			}
		}
	}

	GIVEN("a snapshot file in read mode")
	{
		const char* const path = "test/data/VTFSnapshotFile/atom-lines-1.vtf";

		REQUIRE(boost::filesystem::is_regular_file(path));

		VTFSnapshotFile snapshot(path);

		THEN("`getNumberOfAtoms` returns `28`")
		{
			REQUIRE(snapshot.getNumberOfAtoms() == 28);
		}

		AND_THEN("`isValidAtomID` returns `true` for IDs < 28")
		{
			REQUIRE(snapshot.isValidAtomID(0));
			REQUIRE(snapshot.isValidAtomID(1));
			REQUIRE(snapshot.isValidAtomID(5));
			REQUIRE(snapshot.isValidAtomID(11));
			REQUIRE(snapshot.isValidAtomID(27));
			REQUIRE(!snapshot.isValidAtomID(28));
			REQUIRE(!snapshot.isValidAtomID(123));
		}
	}

	GIVEN(
		"a snapshot file with a malformed atom line: "
		"there is an odd number of option names/values")
	{
		const char* const path =
			"test/data/VTFSnapshotFile/"
			"atom-lines-malformed-odd-number-of-option-names-values.vtf";

		REQUIRE(boost::filesystem::is_regular_file(path));

		THEN("`OpenMPCD::MalformedFileException` is thrown upon construction")
		{
			REQUIRE_THROWS_AS(
				(VTFSnapshotFile(path)),
				OpenMPCD::MalformedFileException);
		}
	}

	GIVEN(
		"a snapshot file with a malformed atom line: "
		"the `atom` keyword appears, but not first, and is still considered "
		"a keyword rather than an option name/value")
	{
		const char* const path =
			"test/data/VTFSnapshotFile/"
			"atom-lines-malformed-atom-keyword-at-wrong-place.vtf";

		REQUIRE(boost::filesystem::is_regular_file(path));

		THEN("`OpenMPCD::MalformedFileException` is thrown upon construction")
		{
			REQUIRE_THROWS_AS(
				(VTFSnapshotFile(path)),
				OpenMPCD::MalformedFileException);
		}
	}

	GIVEN(
		"a snapshot file with a malformed atom line: "
		"there is a comma missing between two `aid-specifier`s")
	{
		const char* const path =
			"test/data/VTFSnapshotFile/"
			"atom-lines-malformed-missing-comma.vtf";

		REQUIRE(boost::filesystem::is_regular_file(path));

		THEN("`OpenMPCD::MalformedFileException` is thrown upon construction")
		{
			REQUIRE_THROWS_AS(
				(VTFSnapshotFile(path)),
				OpenMPCD::MalformedFileException);
		}
	}

	GIVEN(
		"a snapshot file with a malformed atom line: "
		"the `name` property is too long")
	{
		const char* const path =
			"test/data/VTFSnapshotFile/"
			"atom-lines-malformed-name-too-long.vtf";

		REQUIRE(boost::filesystem::is_regular_file(path));

		THEN("`OpenMPCD::MalformedFileException` is thrown upon construction")
		{
			REQUIRE_THROWS_AS(
				(VTFSnapshotFile(path)),
				OpenMPCD::MalformedFileException);
		}
	}

	GIVEN(
		"a snapshot file with a malformed atom line: "
		"the `type` property is too long")
	{
		const char* const path =
			"test/data/VTFSnapshotFile/"
			"atom-lines-malformed-type-too-long.vtf";

		REQUIRE(boost::filesystem::is_regular_file(path));

		THEN("`OpenMPCD::MalformedFileException` is thrown upon construction")
		{
			REQUIRE_THROWS_AS(
				(VTFSnapshotFile(path)),
				OpenMPCD::MalformedFileException);
		}
	}

	GIVEN(
		"a snapshot file with a malformed atom line: "
		"the `radius` property is not a number")
	{
		const char* const path =
			"test/data/VTFSnapshotFile/"
			"atom-lines-malformed-radius-not-a-number.vtf";

		REQUIRE(boost::filesystem::is_regular_file(path));

		THEN("`OpenMPCD::MalformedFileException` is thrown upon construction")
		{
			REQUIRE_THROWS_AS(
				(VTFSnapshotFile(path)),
				OpenMPCD::MalformedFileException);
		}
	}
}


SCENARIO(
	"`OpenMPCD::VTFSnapshotFile::declareAtoms`, "
	"`OpenMPCD::VTFSnapshotFile::getNumberOfAtoms`, "
	"`OpenMPCD::VTFSnapshotFile::isValidAtomID` "
	"with properties",
	"")
{
	struct Helper
	{
		static bool
			testAtomRange(VTFSnapshotFile& snap, const unsigned int rangeID)
		{
			if(rangeID==1)
			{
				if(snap.getAtomProperties(0).name.get() != "testname")
					return false;

				if(snap.getAtomProperties(0).type.is_initialized())
					return false;

				if(snap.getAtomProperties(0).radius.is_initialized())
					return false;
			}
			else if(rangeID==2)
			{
				for(std::size_t a=1; a<=2; ++a)
				{
					if(snap.getAtomProperties(a).name.get() != "testname")
						return false;

					if(snap.getAtomProperties(a).type.get() != "testtype")
						return false;

					if(snap.getAtomProperties(a).radius.is_initialized())
						return false;
				}
			}
			else if(rangeID==2)
			{
				for(std::size_t a=1; a<=2; ++a)
				{
					if(snap.getAtomProperties(a).name.get() != "testname")
						return false;

					if(snap.getAtomProperties(a).type.get() != "testtype")
						return false;

					if(snap.getAtomProperties(a).radius.is_initialized())
						return false;
				}
			}
			else if(rangeID==3)
			{
				for(std::size_t a=3; a<=5; ++a)
				{
					if(snap.getAtomProperties(a).name.get() != "rad")
						return false;

					if(snap.getAtomProperties(a).type.get() != "hasradius")
						return false;

					if(snap.getAtomProperties(a).radius.get() != 1.23)
						return false;
				}
			}
			else if(rangeID==4)
			{
				for(std::size_t a=6; a<=9; ++a)
				{
					if(snap.getAtomProperties(a).name.get() != "rad")
						return false;

					if(snap.getAtomProperties(a).type.get() != "zeroradius")
						return false;

					if(snap.getAtomProperties(a).radius.get() != 0)
						return false;
				}
			}
			else if(rangeID==5)
			{
				for(std::size_t a=10; a<=14; ++a)
				{
					if(snap.getAtomProperties(a).name.is_initialized())
						return false;

					if(snap.getAtomProperties(a).type.get() != "onlytype")
						return false;

					if(snap.getAtomProperties(a).radius.is_initialized())
						return false;
				}
			}
			else if(rangeID==123)
			{
				for(std::size_t a=15; a<=137; ++a)
				{
					if(snap.getAtomProperties(a).name.is_initialized())
						return false;

					if(snap.getAtomProperties(a).type.is_initialized())
						return false;

					if(snap.getAtomProperties(a).radius.is_initialized())
						return false;
				}
			}
			else
			{
				OPENMPCD_THROW(OpenMPCD::Exception, "Unexpected rangeID.");
			}

			return true;
		}
	};

	GIVEN("a snapshot file in write mode")
	{
		static const char* const path =
			"test/data/VTFSnapshotFile/tmp-atom-lines.vtf";

		boost::filesystem::remove(path);
		boost::scoped_ptr<VTFSnapshotFile> snapshot(new VTFSnapshotFile(path));

		REQUIRE(snapshot->isInWriteMode());

		THEN(
			"`declareAtoms` (with atom properties) can be called")
		{
			const std::pair<std::size_t, std::size_t> declaredAtoms1 =
				snapshot->declareAtoms(1, -1, "testname", "");
			const std::pair<std::size_t, std::size_t> declaredAtoms2 =
				snapshot->declareAtoms(2, -1, "testname", "testtype");
			const std::pair<std::size_t, std::size_t> declaredAtoms3 =
				snapshot->declareAtoms(3, 1.23, "rad", "hasradius");
			const std::pair<std::size_t, std::size_t> declaredAtoms4 =
				snapshot->declareAtoms(4, 0, "rad", "zeroradius");
			const std::pair<std::size_t, std::size_t> declaredAtoms5 =
				snapshot->declareAtoms(5, -1, "", "onlytype");
			const std::pair<std::size_t, std::size_t> declaredAtoms6 =
				snapshot->declareAtoms(123);
			const std::pair<std::size_t, std::size_t> declaredAtoms7 =
				snapshot->declareAtoms(0, -1, "foo", "bar");

			AND_THEN("`declareAtoms` returns the right values")
			{
				REQUIRE(declaredAtoms1.first == 0);
				REQUIRE(declaredAtoms1.second == 0);

				REQUIRE(declaredAtoms2.first == 1);
				REQUIRE(declaredAtoms2.second == 2);

				REQUIRE(declaredAtoms3.first == 3);
				REQUIRE(declaredAtoms3.second == 5);

				REQUIRE(declaredAtoms4.first == 6);
				REQUIRE(declaredAtoms4.second == 9);

				REQUIRE(declaredAtoms5.first == 10);
				REQUIRE(declaredAtoms5.second == 14);

				REQUIRE(declaredAtoms6.first == 15);
				REQUIRE(declaredAtoms6.second == 137);

				REQUIRE(declaredAtoms7.first == 0);
				REQUIRE(declaredAtoms7.second == 0);
			}

			AND_THEN("`getNumberOfAtoms` returns `138`")
			{
				REQUIRE(snapshot->getNumberOfAtoms() == 138);
			}

			AND_THEN("`isValidAtomID` returns `true` for IDs < 138")
			{
				REQUIRE(snapshot->isValidAtomID(0));
				REQUIRE(snapshot->isValidAtomID(1));
				REQUIRE(snapshot->isValidAtomID(5));
				REQUIRE(snapshot->isValidAtomID(122));
				REQUIRE(snapshot->isValidAtomID(137));
				REQUIRE(!snapshot->isValidAtomID(138));
				REQUIRE(!snapshot->isValidAtomID(1234));
			}

			AND_THEN("`getAtomProperties` returns the expected properties")
			{
				REQUIRE(Helper::testAtomRange(*snapshot, 1));
				REQUIRE(Helper::testAtomRange(*snapshot, 2));
				REQUIRE(Helper::testAtomRange(*snapshot, 3));
				REQUIRE(Helper::testAtomRange(*snapshot, 4));
				REQUIRE(Helper::testAtomRange(*snapshot, 5));
				REQUIRE(Helper::testAtomRange(*snapshot, 123));
			}

			AND_THEN(
				"`getNumberOfAtoms` on a new instance "
				"returns the correct value, "
				"`isValidAtomID` works as expected, "
				"so does `getAtomProperties`")
			{
				snapshot.reset();
				VTFSnapshotFile readsnap(path);

				REQUIRE(readsnap.isInReadMode());
				REQUIRE(readsnap.getNumberOfAtoms() == 138);

				REQUIRE(readsnap.isValidAtomID(0));
				REQUIRE(readsnap.isValidAtomID(1));
				REQUIRE(readsnap.isValidAtomID(5));
				REQUIRE(readsnap.isValidAtomID(137));
				REQUIRE(!readsnap.isValidAtomID(138));
				REQUIRE(!readsnap.isValidAtomID(1234));

				REQUIRE(Helper::testAtomRange(readsnap, 1));
				REQUIRE(Helper::testAtomRange(readsnap, 2));
				REQUIRE(Helper::testAtomRange(readsnap, 3));
				REQUIRE(Helper::testAtomRange(readsnap, 4));
				REQUIRE(Helper::testAtomRange(readsnap, 5));
				REQUIRE(Helper::testAtomRange(readsnap, 123));
			}
		}
	}

	GIVEN("a snapshot file in read mode, atoms ordered")
	{
		const char* const path =
			"test/data/VTFSnapshotFile/atom-lines-with-properties-1.vtf";

		REQUIRE(boost::filesystem::is_regular_file(path));

		VTFSnapshotFile snapshot(path);

		THEN("`getNumberOfAtoms` returns `138`")
		{
			REQUIRE(snapshot.getNumberOfAtoms() == 138);
		}

		AND_THEN("`isValidAtomID` returns `true` for IDs < 138")
		{
			REQUIRE(snapshot.isValidAtomID(0));
			REQUIRE(snapshot.isValidAtomID(1));
			REQUIRE(snapshot.isValidAtomID(5));
			REQUIRE(snapshot.isValidAtomID(11));
			REQUIRE(snapshot.isValidAtomID(137));
			REQUIRE(!snapshot.isValidAtomID(138));
			REQUIRE(!snapshot.isValidAtomID(1234));
		}

		AND_THEN("`getAtomProperties` works as expected")
		{
			REQUIRE(Helper::testAtomRange(snapshot, 1));
			REQUIRE(Helper::testAtomRange(snapshot, 2));
			REQUIRE(Helper::testAtomRange(snapshot, 3));
			REQUIRE(Helper::testAtomRange(snapshot, 4));
			REQUIRE(Helper::testAtomRange(snapshot, 5));
			REQUIRE(Helper::testAtomRange(snapshot, 123));
		}
	}

	GIVEN("a snapshot file in read mode, atoms unordered")
	{
		const char* const path =
			"test/data/VTFSnapshotFile/atom-lines-with-properties-2.vtf";

		REQUIRE(boost::filesystem::is_regular_file(path));

		VTFSnapshotFile snapshot(path);

		THEN("`getNumberOfAtoms` returns `138`")
		{
			REQUIRE(snapshot.getNumberOfAtoms() == 138);
		}

		AND_THEN("`isValidAtomID` returns `true` for IDs < 138")
		{
			REQUIRE(snapshot.isValidAtomID(0));
			REQUIRE(snapshot.isValidAtomID(1));
			REQUIRE(snapshot.isValidAtomID(5));
			REQUIRE(snapshot.isValidAtomID(11));
			REQUIRE(snapshot.isValidAtomID(137));
			REQUIRE(!snapshot.isValidAtomID(138));
			REQUIRE(!snapshot.isValidAtomID(1234));
		}

		AND_THEN("`getAtomProperties` works as expected")
		{
			REQUIRE(Helper::testAtomRange(snapshot, 1));
			REQUIRE(Helper::testAtomRange(snapshot, 2));
			REQUIRE(Helper::testAtomRange(snapshot, 3));
			REQUIRE(Helper::testAtomRange(snapshot, 4));
			REQUIRE(Helper::testAtomRange(snapshot, 5));
			REQUIRE(Helper::testAtomRange(snapshot, 123));
		}
	}
}


SCENARIO("`OpenMPCD::VTFSnapshotFile::declareBond`", "")
{
	GIVEN("a snapshot file in write mode")
	{
		static const char* const path =
			"test/data/VTFSnapshotFile/tmp-bond-lines.vtf";

		boost::filesystem::remove(path);
		VTFSnapshotFile snapshot(path);

		REQUIRE(snapshot.isInWriteMode());

		WHEN("no atoms have been defined yet")
		{
			THEN("`declareBond` throws `OpenMPCD::OutOfBoundsException`")
			{
				REQUIRE_THROWS_AS(
					snapshot.declareBond(0, 1),
					OpenMPCD::OutOfBoundsException);
			}
		}

		WHEN("only one atom has been defined")
		{
			snapshot.declareAtoms(1);

			THEN(
				"`declareBond` throws `OpenMPCD::OutOfBoundsException` "
				"for invalid atom IDs.")
			{
				REQUIRE_THROWS_AS(
					snapshot.declareBond(0, 1),
					OpenMPCD::OutOfBoundsException);

				REQUIRE_THROWS_AS(
					snapshot.declareBond(1, 0),
					OpenMPCD::OutOfBoundsException);
			}

			THEN(
				"`delcareBond` throws `OpenMPCD::InvalidArgumentException` "
				"if both arguments are the same")
			{
				REQUIRE_THROWS_AS(
					snapshot.declareBond(0, 0),
					OpenMPCD::InvalidArgumentException);
			}
		}

		WHEN("sufficiently many atoms have been defined")
		{
			snapshot.declareAtoms(10);

			THEN(
				"`declareBond` throws `OpenMPCD::OutOfBoundsException` "
				"for invalid atom IDs.")
			{
				REQUIRE_THROWS_AS(
					snapshot.declareBond(0, 10),
					OpenMPCD::OutOfBoundsException);

				REQUIRE_THROWS_AS(
					snapshot.declareBond(20, 0),
					OpenMPCD::OutOfBoundsException);
			}

			THEN(
				"`delcareBond` throws `OpenMPCD::InvalidArgumentException` "
				"if both arguments are the same")
			{
				REQUIRE_THROWS_AS(
					snapshot.declareBond(0, 0),
					OpenMPCD::InvalidArgumentException);
			}

			THEN("`declareBond` can be used to declare bonds")
			{
				snapshot.declareBond(0, 1);
				snapshot.declareBond(0, 5);
				snapshot.declareBond(5, 3);
				snapshot.declareBond(9, 8);
			}

			AND_WHEN("a certain bond has already been defined")
			{
				snapshot.declareBond(0, 5);

				THEN("it cannot be re-declared")
				{
					REQUIRE_THROWS_AS(
						snapshot.declareBond(0, 5),
						OpenMPCD::InvalidArgumentException);

					REQUIRE_THROWS_AS(
						snapshot.declareBond(5, 0),
						OpenMPCD::InvalidArgumentException);
				}
			}
		}
	}

	GIVEN("a snapshot file in read mode")
	{
		const char* const path = "test/data/VTFSnapshotFile/atom-lines-1.vtf";

		REQUIRE(boost::filesystem::is_regular_file(path));

		VTFSnapshotFile snapshot(path);

		REQUIRE(snapshot.isValidAtomID(0));
		REQUIRE(snapshot.isValidAtomID(1));

		THEN("`declareBond` throws `OpenMPCD::InvalidCallException`")
		{
			REQUIRE_THROWS_AS(
				snapshot.declareBond(0, 1),
				OpenMPCD::InvalidCallException);
		}
	}
}


SCENARIO(
	"`OpenMPCD::VTFSnapshotFile::getBonds`",
	"")
{
	GIVEN("a snapshot file in write mode")
	{
		static const char* const path =
			"test/data/VTFSnapshotFile/tmp-bond-lines.vtf";

		boost::filesystem::remove(path);
		VTFSnapshotFile snapshot(path);

		REQUIRE(snapshot.isInWriteMode());

		WHEN("no bonds have been defined yet")
		{
			THEN("`getBonds` returns an empty set")
			{
				REQUIRE(snapshot.getBonds().empty());
			}
		}

		WHEN("some bonds have been defined")
		{
			snapshot.declareAtoms(10);

			snapshot.declareBond(0, 1);
			snapshot.declareBond(9, 7);
			snapshot.declareBond(0, 2);
			snapshot.declareBond(2, 5);

			THEN("`getBonds` returns those bonds")
			{
				typedef std::set<std::pair<std::size_t, std::size_t> > Set;
				const Set bonds = snapshot.getBonds();

				REQUIRE(bonds.size() == 4);

				Set::const_iterator it = bonds.begin();
				REQUIRE(it != bonds.end());

				REQUIRE(it->first == 0);
				REQUIRE(it->second == 1);
				REQUIRE(++it != bonds.end());

				REQUIRE(it->first == 0);
				REQUIRE(it->second == 2);
				REQUIRE(++it != bonds.end());

				REQUIRE(it->first == 2);
				REQUIRE(it->second == 5);
				REQUIRE(++it != bonds.end());

				REQUIRE(it->first == 7);
				REQUIRE(it->second == 9);
				REQUIRE(++it == bonds.end());
			}
		}
	}

	GIVEN("a snapshot file in read mode")
	{
		const char* const path = "test/data/VTFSnapshotFile/bond-lines-1.vtf";

		REQUIRE(boost::filesystem::is_regular_file(path));

		VTFSnapshotFile snapshot(path);

		THEN("`getBonds` returns the bonds declared in the file")
		{
			typedef std::set<std::pair<std::size_t, std::size_t> > Set;
			const Set bonds = snapshot.getBonds();

			REQUIRE(bonds.size() == 14);

			Set::const_iterator it = bonds.begin();
			REQUIRE(it != bonds.end());

			REQUIRE(it->first == 0);
			REQUIRE(it->second == 1);
			REQUIRE(++it != bonds.end());

			REQUIRE(it->first == 1);
			REQUIRE(it->second == 3);
			REQUIRE(++it != bonds.end());

			REQUIRE(it->first == 1);
			REQUIRE(it->second == 5);
			REQUIRE(++it != bonds.end());

			REQUIRE(it->first == 2);
			REQUIRE(it->second == 3);
			REQUIRE(++it != bonds.end());

			REQUIRE(it->first == 9);
			REQUIRE(it->second == 10);
			REQUIRE(++it != bonds.end());

			REQUIRE(it->first == 11);
			REQUIRE(it->second == 12);
			REQUIRE(++it != bonds.end());

			REQUIRE(it->first == 13);
			REQUIRE(it->second == 14);
			REQUIRE(++it != bonds.end());

			REQUIRE(it->first == 14);
			REQUIRE(it->second == 15);
			REQUIRE(++it != bonds.end());

			REQUIRE(it->first == 15);
			REQUIRE(it->second == 16);
			REQUIRE(++it != bonds.end());

			REQUIRE(it->first == 20);
			REQUIRE(it->second == 21);
			REQUIRE(++it != bonds.end());

			REQUIRE(it->first == 21);
			REQUIRE(it->second == 22);
			REQUIRE(++it != bonds.end());

			REQUIRE(it->first == 26);
			REQUIRE(it->second == 27);
			REQUIRE(++it != bonds.end());

			REQUIRE(it->first == 27);
			REQUIRE(it->second == 28);
			REQUIRE(++it != bonds.end());

			REQUIRE(it->first == 28);
			REQUIRE(it->second == 29);
			REQUIRE(++it == bonds.end());
		}
	}
}


SCENARIO(
	"`OpenMPCD::VTFSnapshotFile::hasBond`",
	"")
{
	GIVEN("a snapshot file in write mode")
	{
		static const char* const path =
			"test/data/VTFSnapshotFile/tmp-bond-lines.vtf";

		boost::filesystem::remove(path);
		VTFSnapshotFile snapshot(path);

		REQUIRE(snapshot.isInWriteMode());

		WHEN("no atoms have been defined yet")
		{
			THEN("`hasBonds` throws `OpenMPCD::InvalidArgumentException`")
			{
				REQUIRE_THROWS_AS(
					snapshot.hasBond(0, 1),
					OpenMPCD::OutOfBoundsException);
			}
		}

		WHEN("some bonds have been defined")
		{
			snapshot.declareAtoms(10);

			snapshot.declareBond(0, 1);
			snapshot.declareBond(9, 7);
			snapshot.declareBond(0, 2);
			snapshot.declareBond(2, 5);

			THEN(
				"`hasBonds` throws `OpenMPCD::InvalidArgumentException` "
				"if the two arguments are the same")
			{
				REQUIRE_THROWS_AS(
					snapshot.hasBond(0, 0),
					OpenMPCD::InvalidArgumentException);

				REQUIRE_THROWS_AS(
					snapshot.hasBond(3, 3),
					OpenMPCD::InvalidArgumentException);
			}

			THEN("`hasBonds` returns correct results")
			{
				REQUIRE(snapshot.hasBond(0, 1));
				REQUIRE(snapshot.hasBond(1, 0));

				REQUIRE(snapshot.hasBond(9, 7));
				REQUIRE(snapshot.hasBond(7, 9));

				REQUIRE(snapshot.hasBond(0, 2));
				REQUIRE(snapshot.hasBond(2, 0));

				REQUIRE(!snapshot.hasBond(2, 3));
				REQUIRE(!snapshot.hasBond(3, 2));

				REQUIRE(!snapshot.hasBond(4, 5));
			}
		}
	}

	GIVEN("a snapshot file in read mode")
	{
		const char* const path = "test/data/VTFSnapshotFile/bond-lines-1.vtf";

		REQUIRE(boost::filesystem::is_regular_file(path));

		VTFSnapshotFile snapshot(path);

		THEN("`hasBond` returns correct results")
		{
			REQUIRE(snapshot.hasBond(0, 1));
			REQUIRE(snapshot.hasBond(1, 0));

			REQUIRE(snapshot.hasBond(1, 3));
			REQUIRE(snapshot.hasBond(3, 1));

			REQUIRE(snapshot.hasBond(13, 14));

			REQUIRE(snapshot.hasBond(14, 15));
		}
	}
}


SCENARIO(
	"`OpenMPCD::VTFSnapshotFile::writeTimestepBlock`, "
	"`OpenMPCD::VTFSnapshotFile::readTimestepBlock`",
	"")
{
	using OpenMPCD::FP;

	GIVEN("a snapshot file in write mode")
	{
		static const char* const path =
			"test/data/VTFSnapshotFile/tmp-timestep-block.vtf";

		static const std::size_t atomCount = 100;

		boost::filesystem::remove(path);
		boost::scoped_ptr<VTFSnapshotFile> snapshot(new VTFSnapshotFile(path));

		REQUIRE(snapshot->isInWriteMode());

		snapshot->declareAtoms(atomCount);

		THEN("`readTimestepBlock` throws `InvalidCallException`")
		{
			FP tmp;
			REQUIRE_THROWS_AS(
				snapshot->readTimestepBlock(&tmp),
				OpenMPCD::InvalidCallException);
		}

		THEN("`writeTimestepBlock` throws `NULLPointerException` if passed 0")
		{
			REQUIRE_THROWS_AS(
				snapshot->writeTimestepBlock(0),
				OpenMPCD::NULLPointerException);
		}

		THEN(
			"`writeTimestepBlock` will work if called once, "
			"without velocities")
		{
			FP positions[3 * atomCount];

			for(std::size_t a = 0; a < atomCount; ++a)
			{
				for(std::size_t coord = 0; coord < 3; ++coord)
					positions[3 * a + coord] = (3 * a + coord) / 3.0;
			}

			snapshot->writeTimestepBlock(positions);

			AND_THEN(
					"`readTimestepBlock` on a new instance reads the "
					"correct values, can be called multiple times")
			{
				snapshot.reset();
				VTFSnapshotFile readsnap(path);

				REQUIRE(readsnap.isInReadMode());
				REQUIRE(readsnap.getNumberOfAtoms() == atomCount);

				FP readPositions[3 * atomCount];

				REQUIRE(readsnap.readTimestepBlock(readPositions));

				for(std::size_t a = 0; a < atomCount; ++a)
				{
					for(std::size_t coord = 0; coord < 3; ++coord)
					{
						REQUIRE(
							readPositions[3 * a + coord]
							==
							positions[3 * a + coord]);
					}
				}

				REQUIRE_FALSE(readsnap.readTimestepBlock(readPositions));

				for(std::size_t a = 0; a < atomCount; ++a)
				{
					for(std::size_t coord = 0; coord < 3; ++coord)
					{
						REQUIRE(
							readPositions[3 * a + coord]
							==
							positions[3 * a + coord]);
					}
				}

				REQUIRE_FALSE(readsnap.readTimestepBlock(readPositions));
				REQUIRE_FALSE(readsnap.readTimestepBlock(0));
			}

			AND_THEN(
					"`readTimestepBlock` on a new instance reads the "
					"correct values, can be called multiple times, "
					"reads no velocities")
			{
				snapshot.reset();
				VTFSnapshotFile readsnap(path);

				REQUIRE(readsnap.isInReadMode());
				REQUIRE(readsnap.getNumberOfAtoms() == atomCount);

				FP readPositions[3 * atomCount];
				FP readVelocities[3 * atomCount];

				REQUIRE(
					readsnap.readTimestepBlock(
						readPositions, readVelocities));

				for(std::size_t a = 0; a < atomCount; ++a)
				{
					for(std::size_t coord = 0; coord < 3; ++coord)
					{
						REQUIRE(
							readPositions[3 * a + coord]
							==
							positions[3 * a + coord]);
					}
				}

				REQUIRE_FALSE(
					readsnap.readTimestepBlock(readPositions, readVelocities));

				for(std::size_t a = 0; a < atomCount; ++a)
				{
					for(std::size_t coord = 0; coord < 3; ++coord)
					{
						REQUIRE(
							readPositions[3 * a + coord]
							==
							positions[3 * a + coord]);
					}
				}

				REQUIRE_FALSE(
					readsnap.readTimestepBlock(readPositions, readVelocities));
				REQUIRE_FALSE(readsnap.readTimestepBlock(0));
			}

			AND_THEN(
					"`readTimestepBlock` on a new instance reads the "
					"correct values, can be called multiple times, "
					"reads no velocities, stores `velocitiesEncountered=false`")
			{
				snapshot.reset();
				VTFSnapshotFile readsnap(path);

				REQUIRE(readsnap.isInReadMode());
				REQUIRE(readsnap.getNumberOfAtoms() == atomCount);

				FP readPositions[3 * atomCount];
				FP readVelocities[3 * atomCount];
				bool velocitiesEncountered = true;

				REQUIRE(
					readsnap.readTimestepBlock(
						readPositions, readVelocities, &velocitiesEncountered));
				REQUIRE_FALSE(velocitiesEncountered);

				for(std::size_t a = 0; a < atomCount; ++a)
				{
					for(std::size_t coord = 0; coord < 3; ++coord)
					{
						REQUIRE(
							readPositions[3 * a + coord]
							==
							positions[3 * a + coord]);
					}
				}

				REQUIRE_FALSE(
					readsnap.readTimestepBlock(
						readPositions, readVelocities, &velocitiesEncountered));

				for(std::size_t a = 0; a < atomCount; ++a)
				{
					for(std::size_t coord = 0; coord < 3; ++coord)
					{
						REQUIRE(
							readPositions[3 * a + coord]
							==
							positions[3 * a + coord]);
					}
				}

				REQUIRE_FALSE(
					readsnap.readTimestepBlock(
						readPositions, readVelocities, &velocitiesEncountered));
				REQUIRE_FALSE(readsnap.readTimestepBlock(0));
			}
		}


		THEN(
			"`writeTimestepBlock` will work if called twice, "
			"without velocities")
		{
			FP positions[3 * atomCount];

			for(std::size_t a = 0; a < atomCount; ++a)
			{
				for(std::size_t coord = 0; coord < 3; ++coord)
					positions[3 * a + coord] = (3 * a + coord) / 3.0;
			}

			snapshot->writeTimestepBlock(positions);

			for(std::size_t i = 0; i < 3 * atomCount; ++i)
				positions[i] *= -1;

			snapshot->writeTimestepBlock(positions);

			AND_THEN(
					"`readTimestepBlock` on a new instance reads the "
					"correct values, can be called multiple times")
			{
				snapshot.reset();
				VTFSnapshotFile readsnap(path);

				REQUIRE(readsnap.isInReadMode());
				REQUIRE(readsnap.getNumberOfAtoms() == atomCount);

				FP readPositions[3 * atomCount];

				REQUIRE(readsnap.readTimestepBlock(readPositions));

				for(std::size_t a = 0; a < atomCount; ++a)
				{
					for(std::size_t coord = 0; coord < 3; ++coord)
					{
						REQUIRE(
							readPositions[3 * a + coord]
							==
							-1 * positions[3 * a + coord]);
					}
				}

				REQUIRE(readsnap.readTimestepBlock(readPositions));

				for(std::size_t a = 0; a < atomCount; ++a)
				{
					for(std::size_t coord = 0; coord < 3; ++coord)
					{
						REQUIRE(
							readPositions[3 * a + coord]
							==
							positions[3 * a + coord]);
					}
				}

				REQUIRE_FALSE(readsnap.readTimestepBlock(readPositions));
				REQUIRE_FALSE(readsnap.readTimestepBlock(0));
			}
		}



		THEN(
			"`writeTimestepBlock` will work if called once, "
			"with velocities")
		{
			FP positions[3 * atomCount];
			FP velocities[3 * atomCount];

			for(std::size_t a = 0; a < atomCount; ++a)
			{
				for(std::size_t coord = 0; coord < 3; ++coord)
				{
					positions[3 * a + coord] = (3 * a + coord) / 3.0;
					velocities[3 * a + coord] = - positions[3 * a + coord] + 1;
				}
			}

			snapshot->writeTimestepBlock(positions, velocities);

			AND_THEN(
					"`readTimestepBlock` on a new instance reads the "
					"correct values, can be called multiple times")
			{
				snapshot.reset();
				VTFSnapshotFile readsnap(path);

				REQUIRE(readsnap.isInReadMode());
				REQUIRE(readsnap.getNumberOfAtoms() == atomCount);

				FP readPositions[3 * atomCount];
				FP readVelocities[3 * atomCount];
				bool velocitiesEncountered;

				REQUIRE(
					readsnap.readTimestepBlock(
						readPositions, readVelocities, &velocitiesEncountered));

				for(std::size_t a = 0; a < atomCount; ++a)
				{
					for(std::size_t coord = 0; coord < 3; ++coord)
					{
						REQUIRE(
							readPositions[3 * a + coord]
							==
							positions[3 * a + coord]);
						REQUIRE(
							readVelocities[3 * a + coord]
							==
							velocities[3 * a + coord]);
					}
				}

				REQUIRE_FALSE(
					readsnap.readTimestepBlock(
						readPositions, readVelocities, &velocitiesEncountered));

				for(std::size_t a = 0; a < atomCount; ++a)
				{
					for(std::size_t coord = 0; coord < 3; ++coord)
					{
						REQUIRE(
							readPositions[3 * a + coord]
							==
							positions[3 * a + coord]);
						REQUIRE(
							readVelocities[3 * a + coord]
							==
							velocities[3 * a + coord]);
					}
				}

				REQUIRE_FALSE(readsnap.readTimestepBlock(readPositions));
				REQUIRE_FALSE(readsnap.readTimestepBlock(0));
			}
		}


		THEN(
			"`writeTimestepBlock` will work if called twice, "
			"with velocities")
		{
			FP positions[3 * atomCount];
			FP velocities[3 * atomCount];

			for(std::size_t a = 0; a < atomCount; ++a)
			{
				for(std::size_t coord = 0; coord < 3; ++coord)
				{
					positions[3 * a + coord] = (3 * a + coord) / 3.0;
					velocities[3 * a + coord] = - positions[3 * a + coord] + 1;
				}
			}

			snapshot->writeTimestepBlock(positions, velocities);

			for(std::size_t i = 0; i < 3 * atomCount; ++i)
			{
				positions[i] *= -1;
				velocities[i] *= -1;
			}

			snapshot->writeTimestepBlock(positions, velocities);

			AND_THEN(
					"`readTimestepBlock` on a new instance reads the "
					"correct values, can be called multiple times")
			{
				snapshot.reset();
				VTFSnapshotFile readsnap(path);

				REQUIRE(readsnap.isInReadMode());
				REQUIRE(readsnap.getNumberOfAtoms() == atomCount);

				FP readPositions[3 * atomCount];
				FP readVelocities[3 * atomCount];
				bool velocitiesEncountered = false;

				REQUIRE(
					readsnap.readTimestepBlock(
						readPositions, readVelocities, &velocitiesEncountered));
				REQUIRE(velocitiesEncountered);

				for(std::size_t a = 0; a < atomCount; ++a)
				{
					for(std::size_t coord = 0; coord < 3; ++coord)
					{
						REQUIRE(
							readPositions[3 * a + coord]
							==
							-1 * positions[3 * a + coord]);
						REQUIRE(
							readVelocities[3 * a + coord]
							==
							-1 * velocities[3 * a + coord]);
					}
				}

				velocitiesEncountered = false;
				REQUIRE(
					readsnap.readTimestepBlock(
						readPositions, readVelocities, &velocitiesEncountered));
				REQUIRE(velocitiesEncountered);

				for(std::size_t a = 0; a < atomCount; ++a)
				{
					for(std::size_t coord = 0; coord < 3; ++coord)
					{
						REQUIRE(
							readPositions[3 * a + coord]
							==
							positions[3 * a + coord]);
						REQUIRE(
							readVelocities[3 * a + coord]
							==
							velocities[3 * a + coord]);
					}
				}

				REQUIRE_FALSE(readsnap.readTimestepBlock(readPositions));
				REQUIRE_FALSE(readsnap.readTimestepBlock(0));
			}
		}
	}

	GIVEN("a snapshot file in read mode, timestep blocks ordered")
	{
		const char* const path =
			"test/data/VTFSnapshotFile/timestep-blocks-ordered.vtf";

		static const std::size_t atomCount = 50;

		REQUIRE(boost::filesystem::is_regular_file(path));

		VTFSnapshotFile snapshot(path);

		THEN("`writeTimestepBlock` throws `InvalidCallException`")
		{
			FP tmp;
			REQUIRE_THROWS_AS(
				snapshot.writeTimestepBlock(&tmp),
				OpenMPCD::InvalidCallException);
		}

		THEN("`getNumberOfAtoms` returns `50`")
		{
			REQUIRE(snapshot.getNumberOfAtoms() == atomCount);
		}

		THEN("`readTimestepBlock` returns `true` and the appropriate values")
		{
			FP positions[atomCount * 3];

			REQUIRE(snapshot.readTimestepBlock(positions));

			for(std::size_t a = 0; a < atomCount; ++a)
			{
				REQUIRE(positions[a * 3 + 0] == Approx(-1.0 * a / 2.0));
				REQUIRE(positions[a * 3 + 1] == Approx(a / 2.0));
				REQUIRE(positions[a * 3 + 2] == Approx(a));
			}

			REQUIRE(snapshot.readTimestepBlock(0));

			REQUIRE(snapshot.readTimestepBlock(positions));

			for(std::size_t a = 0; a < atomCount; ++a)
			{
				REQUIRE(positions[a * 3 + 0] == Approx(a / 3.0));
				REQUIRE(positions[a * 3 + 1] == Approx(3 * a / 2.0));
				REQUIRE(positions[a * 3 + 2] == Approx(-2.0 * a / 3.0));
			}
		}
	}

	GIVEN(
		"a snapshot file in read mode, timestep blocks ordered, "
		"with velocities")
	{
		const char* const path =
			"test/data/VTFSnapshotFile/timestep-blocks-ordered-velocities.vtf";

		static const std::size_t atomCount = 50;

		REQUIRE(boost::filesystem::is_regular_file(path));

		VTFSnapshotFile snapshot(path);

		THEN("`writeTimestepBlock` throws `InvalidCallException`")
		{
			FP tmp;
			REQUIRE_THROWS_AS(
				snapshot.writeTimestepBlock(&tmp),
				OpenMPCD::InvalidCallException);
		}

		THEN("`getNumberOfAtoms` returns `50`")
		{
			REQUIRE(snapshot.getNumberOfAtoms() == atomCount);
		}

		THEN("`readTimestepBlock` returns `true` and the appropriate values")
		{
			FP positions[atomCount * 3];
			FP velocities[atomCount * 3];
			bool velocitiesEncountered = false;

			REQUIRE(
				snapshot.readTimestepBlock(
					positions, velocities, &velocitiesEncountered));
			REQUIRE(velocitiesEncountered);

			for(std::size_t a = 0; a < atomCount; ++a)
			{
				REQUIRE(positions[a * 3 + 0] == Approx(-1.0 * a / 2.0));
				REQUIRE(positions[a * 3 + 1] == Approx(a / 2.0));
				REQUIRE(positions[a * 3 + 2] == Approx(a));

				REQUIRE(velocities[a * 3 + 0] == Approx(a / 4.0));
				REQUIRE(velocities[a * 3 + 1] == Approx(a / 5.0));
				REQUIRE(velocities[a * 3 + 2] == Approx(a / 6.0));
			}


			REQUIRE(
				snapshot.readTimestepBlock(NULL, NULL, &velocitiesEncountered));
			REQUIRE_FALSE(velocitiesEncountered);


			REQUIRE(
				snapshot.readTimestepBlock(
					positions, NULL, &velocitiesEncountered));
			REQUIRE(velocitiesEncountered);

			for(std::size_t a = 0; a < atomCount; ++a)
			{
				REQUIRE(positions[a * 3 + 0] == Approx(a / 3.0));
				REQUIRE(positions[a * 3 + 1] == Approx(3 * a / 2.0));
				REQUIRE(positions[a * 3 + 2] == Approx(-2.0 * a / 3.0));

				//old velocities
				REQUIRE(velocities[a * 3 + 0] == Approx(a / 4.0));
				REQUIRE(velocities[a * 3 + 1] == Approx(a / 5.0));
				REQUIRE(velocities[a * 3 + 2] == Approx(a / 6.0));
			}


			velocitiesEncountered = false;
			REQUIRE(
				snapshot.readTimestepBlock(
					positions, velocities, &velocitiesEncountered));
			REQUIRE(velocitiesEncountered);

			for(std::size_t a = 0; a < atomCount; ++a)
			{
				REQUIRE(positions[a * 3 + 0] == Approx(a / 3.0));
				REQUIRE(positions[a * 3 + 1] == Approx(3 * a / 2.0));
				REQUIRE(positions[a * 3 + 2] == Approx(-2.0 * a / 3.0));

				//new velocities
				REQUIRE(velocities[a * 3 + 0] == Approx(a));
				REQUIRE(velocities[a * 3 + 1] == 0);
				REQUIRE(velocities[a * 3 + 2] == 1);
			}
		}
	}

	GIVEN(
		"a snapshot file in read mode, timestep blocks ordered, "
		"with velocities")
	{
		const char* const path =
			"test/data/VTFSnapshotFile/timestep-blocks-ordered-velocities.vtf";

		static const std::size_t atomCount = 50;

		REQUIRE(boost::filesystem::is_regular_file(path));

		VTFSnapshotFile snapshot(path);

		THEN("`getNumberOfAtoms` returns `50`")
		{
			REQUIRE(snapshot.getNumberOfAtoms() == atomCount);
		}

		THEN(
			"`readTimestepBlock` leaves velocities unchanged if none are "
			"encountered")
		{
			FP velocities[atomCount * 3];
			bool velocitiesEncountered = false;

			REQUIRE(
				snapshot.readTimestepBlock(
					NULL, velocities, &velocitiesEncountered));
			REQUIRE(velocitiesEncountered);

			for(std::size_t a = 0; a < atomCount; ++a)
			{
				REQUIRE(velocities[a * 3 + 0] == Approx(a / 4.0));
				REQUIRE(velocities[a * 3 + 1] == Approx(a / 5.0));
				REQUIRE(velocities[a * 3 + 2] == Approx(a / 6.0));
			}


			REQUIRE(
				snapshot.readTimestepBlock(
					NULL, velocities, &velocitiesEncountered));
			REQUIRE_FALSE(velocitiesEncountered);
			for(std::size_t a = 0; a < atomCount; ++a)
			{
				REQUIRE(velocities[a * 3 + 0] == Approx(a / 4.0));
				REQUIRE(velocities[a * 3 + 1] == Approx(a / 5.0));
				REQUIRE(velocities[a * 3 + 2] == Approx(a / 6.0));
			}


			REQUIRE(
				snapshot.readTimestepBlock(
					NULL, NULL, &velocitiesEncountered));
			REQUIRE(velocitiesEncountered);


			velocitiesEncountered = false;
			REQUIRE(
				snapshot.readTimestepBlock(
					NULL, velocities, &velocitiesEncountered));
			REQUIRE(velocitiesEncountered);

			for(std::size_t a = 0; a < atomCount; ++a)
			{
				//new velocities
				REQUIRE(velocities[a * 3 + 0] == Approx(a));
				REQUIRE(velocities[a * 3 + 1] == 0);
				REQUIRE(velocities[a * 3 + 2] == 1);
			}
		}
	}

	GIVEN("a snapshot file in read mode, timestep block indexed")
	{
		const char* const path =
			"test/data/VTFSnapshotFile/timestep-blocks-indexed.vtf";

		REQUIRE(boost::filesystem::is_regular_file(path));

		VTFSnapshotFile snapshot(path);

		THEN("`readTimestepBlock` throws `UnimplementedException`")
		{
			FP tmp;
			REQUIRE_THROWS_AS(
				snapshot.readTimestepBlock(&tmp),
				OpenMPCD::UnimplementedException);
		}
	}

	GIVEN("a snapshot file in read mode, too few positions")
	{
		const char* const path =
			"test/data/VTFSnapshotFile/timestep-blocks-too-few-positions.vtf";

		REQUIRE(boost::filesystem::is_regular_file(path));

		VTFSnapshotFile snapshot(path);

		THEN("`readTimestepBlock` throws `MalformedFileException`")
		{
			REQUIRE_THROWS_AS(
				snapshot.readTimestepBlock(0),
				OpenMPCD::MalformedFileException);
		}
	}

	GIVEN("a snapshot file in read mode, partial velocity information")
	{
		const char* const path =
			"test/data/VTFSnapshotFile/"
			"timestep-blocks-partial-velocity-information.vtf";

		REQUIRE(boost::filesystem::is_regular_file(path));

		VTFSnapshotFile snapshot(path);

		THEN("`readTimestepBlock` throws `MalformedFileException`")
		{
			REQUIRE_THROWS_AS(
				snapshot.readTimestepBlock(NULL),
				OpenMPCD::MalformedFileException);
		}
		THEN("`readTimestepBlock` throws `MalformedFileException`")
		{
			boost::scoped_array<FP> pos(new FP[3*snapshot.getNumberOfAtoms()]);
			boost::scoped_array<FP> vel(new FP[3*snapshot.getNumberOfAtoms()]);

			REQUIRE_THROWS_AS(
				snapshot.readTimestepBlock(pos.get(), vel.get()),
				OpenMPCD::MalformedFileException);
		}
		THEN("`readTimestepBlock` throws `MalformedFileException`")
		{
			boost::scoped_array<FP> pos(new FP[3*snapshot.getNumberOfAtoms()]);
			boost::scoped_array<FP> vel(new FP[3*snapshot.getNumberOfAtoms()]);
			bool b;

			REQUIRE_THROWS_AS(
				snapshot.readTimestepBlock(pos.get(), vel.get(), &b),
				OpenMPCD::MalformedFileException);
		}
	}
}
