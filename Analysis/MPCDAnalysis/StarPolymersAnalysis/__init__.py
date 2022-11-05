from .Registry import registry

from .Acylindricity import Acylindricity
from .Asphericity import Asphericity
from .EckartAngularVelocityVector import EckartAngularVelocityVector
from .GyrationTensor import GyrationTensor
from .MagneticClusterCount import MagneticClusterCount
from .OrientationAngles import OrientationAngles
from .PotentialEnergy import PotentialEnergy
from .RadiusOfGyration import RadiusOfGyration
from .RelativeShapeAnisotropy import RelativeShapeAnisotropy
from .RotationFrequencyVector import RotationFrequencyVector

registry.register(PotentialEnergy)

registry.register(Acylindricity)
registry.register(Asphericity)
registry.register(EckartAngularVelocityVector)
registry.register(GyrationTensor)
registry.register(MagneticClusterCount)
registry.register(OrientationAngles)
registry.register(RadiusOfGyration)
registry.register(RelativeShapeAnisotropy)
registry.register(RotationFrequencyVector)