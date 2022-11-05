/**
 * @file
 * Tests `OpenMPCD::CUDA::Random::Distributions::Gamma`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Random/Distributions/Gamma.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/Random/Generators/Philox4x32-10.hpp>

#include <ctime>


static const float parameterCombinations[][2] =
	{
			{0.5f, 0.5f},
			{0.5f, 1.0f},
			{0.5f, 1.5f},

			{1.0f, 0.5f},
			{1.0f, 1.0f},
			{1.0f, 2.0f},

			{2.0f, 0.5f},
			{2.0f, 1.0f},
			{2.0f, 2.0f},
	};
static const std::size_t parameterCombinationCount =
	sizeof(parameterCombinations) / sizeof(parameterCombinations[0]);



static const std::size_t expectedValuesCount = 10;

static const float expected_float_array[][expectedValuesCount] =
{
	{
		0.17284218966960906982421875,
		0.561757981777191162109375,
		0.214486777782440185546875,
		0.21730460226535797119140625,
		0.07157890498638153076171875,
		0.050144903361797332763671875,
		0.1651889979839324951171875,
		0.294597327709197998046875,
		0.00032216167892329394817352294921875,
		0.0029342793859541416168212890625
	},
	{
		0.3456843793392181396484375,
		1.12351596355438232421875,
		0.42897355556488037109375,
		0.4346092045307159423828125,
		0.1431578099727630615234375,
		0.10028980672359466552734375,
		0.330377995967864990234375,
		0.58919465541839599609375,
		0.0006443233578465878963470458984375,
		0.005868558771908283233642578125
	},
	{
		0.518526554107666015625,
		1.6852741241455078125,
		0.64346039295196533203125,
		0.65191376209259033203125,
		0.2147367298603057861328125,
		0.15043471753597259521484375,
		0.495567023754119873046875,
		0.88379204273223876953125,
		0.0009664849494583904743194580078125,
		0.008802838623523712158203125
	},

	{
		0.147418081760406494140625,
		0.600519835948944091796875,
		0.02027820050716400146484375,
		0.4706387221813201904296875,
		0.3432367742061614990234375,
		0.51743495464324951171875,
		1.18467617034912109375,
		1.3358504772186279296875,
		0.22230182588100433349609375,
		0.4131809175014495849609375,
	},
	{
		0.29483616352081298828125,
		1.20103967189788818359375,
		0.0405564010143280029296875,
		0.941277444362640380859375,
		0.686473548412322998046875,
		1.0348699092864990234375,
		2.3693523406982421875,
		2.671700954437255859375,
		0.4446036517620086669921875,
		0.826361835002899169921875
	},
	{
		0.5896723270416259765625,
		2.4020793437957763671875,
		0.081112802028656005859375,
		1.88255488872528076171875,
		1.37294709682464599609375,
		2.069739818572998046875,
		4.738704681396484375,
		5.34340190887451171875,
		0.889207303524017333984375,
		1.65272367000579833984375,
	},

	{
		0.510695397853851318359375,
		1.2252576351165771484375,
		0.1950580775737762451171875,
		1.04123127460479736328125,
		0.8489358425140380859375,
		1.10869610309600830078125,
		1.97255766391754150390625,
		2.1528341770172119140625,
		0.6491420269012451171875,
		0.956254303455352783203125
	},
	{
		1.02139079570770263671875,
		2.450515270233154296875,
		0.390116155147552490234375,
		2.0824625492095947265625,
		1.697871685028076171875,
		2.2173922061920166015625,
		3.9451153278350830078125,
		4.305668354034423828125,
		1.298284053802490234375,
		1.91250860691070556640625
	},
	{
		2.0427815914154052734375,
		4.90103054046630859375,
		0.78023231029510498046875,
		4.164925098419189453125,
		3.39574337005615234375,
		4.434784412384033203125,
		7.890230655670166015625,
		8.61133670806884765625,
		2.59656810760498046875,
		3.8250172138214111328125
	}
};
static const double expected_double_array[][expectedValuesCount] =
{
	{
		0.2084953414425021467337728608981706202030181884765625,
		0.020987937272067581917500689314692863263189792633056640625,
		0.06415826977411560883002294985999469645321369171142578125,
		0.060049390478164176021547149275647825561463832855224609375,
		0.1717998058300243824358943811603239737451076507568359375,
		4.6472564559006150565652337736111121557769365608692169189453125e-05,
		0.2070811331192717041904671759766642935574054718017578125,
		0.46258149115654012550891138744191266596317291259765625,
		0.000104919764604050554687546259469144160902942530810832977294921875,
		0.1402216058331371417722266414784826338291168212890625
	},
	{
		0.416990682885004293467545721796341240406036376953125,
		0.04197587454413516383500137862938572652637958526611328125,
		0.1283165395482312176600458997199893929064273834228515625,
		0.12009878095632835204309429855129565112292766571044921875,
		0.343599611660048764871788762320647947490215301513671875,
		9.294512911801230113130467547222224311553873121738433837890625e-05,
		0.414162266238543408380934351953328587114810943603515625,
		0.9251629823130802510178227748838253319263458251953125,
		0.00020983952920810110937509251893828832180588506162166595458984375,
		0.280443211666274283544453282956965267658233642578125
	},
	{
		0.6254860243275064402013185826945118606090545654296875,
		0.06296381181620273881360816403685021214187145233154296875,
		0.1924748093223468126122810417655273340642452239990234375,
		0.1801481714344925488813231595486286096274852752685546875,
		0.51539941749007300852980506533640436828136444091796875,
		0.00013941769367701843814442985713952793958014808595180511474609375,
		0.62124339935781514032697714355890639126300811767578125,
		1.387744473469620043459826774778775870800018310546875,
		0.0003147592938121516505101116223386270576156675815582275390625,
		0.420664817499411369805528693177620880305767059326171875,
	},

	{
		0.1603854746200568393010854606473003514111042022705078125,
		0.310750315644495012090686714145704172551631927490234375,
		0.38124622593505830270288470273953862488269805908203125,
		1.747056867995450790687073094886727631092071533203125,
		0.328476432187197964207570066719199530780315399169921875,
		1.3191316827452796989206262878724373877048492431640625,
		0.51262564078946148971027696461533196270465850830078125,
		1.437794083141590295582545877550728619098663330078125,
		0.309067729523902101629317940023611299693584442138671875,
		0.32570039747417667275186659026076085865497589111328125,
	},
	{
		0.320770949240113678602170921294600702822208404541015625,
		0.62150063128899002418137342829140834510326385498046875,
		0.7624924518701166054057694054790772497653961181640625,
		3.49411373599090158137414618977345526218414306640625,
		0.65695286437439592841514013343839906156063079833984375,
		2.638263365490559397841252575744874775409698486328125,
		1.0252512815789229794205539292306639254093170166015625,
		2.87558816628318059116509175510145723819732666015625,
		0.61813545904780420325863588004722259938716888427734375,
		0.6514007949483533455037331805215217173099517822265625
	},
	{
		0.64154189848022735720434184258920140564441680908203125,
		1.2430012625779800483627468565828166902065277099609375,
		1.524984903740233210811538810958154499530792236328125,
		6.9882274719818031627482923795469105243682861328125,
		1.3139057287487918568302802668767981231212615966796875,
		5.27652673098111879568250515148974955081939697265625,
		2.050502563157845958841107858461327850818634033203125,
		5.7511763325663611823301835102029144763946533203125,
		1.2362709180956084065172717600944451987743377685546875,
		1.302801589896706691007466361043043434619903564453125
	},

	{
		0.535869590573458953741692312178201973438262939453125,
		0.79731989796225188893430413372698239982128143310546875,
		0.90784171430118154599853141917265020310878753662109375,
		2.6263262656037085207572090439498424530029296875,
		0.82564009175355124181550081630120985209941864013671875,
		2.133085771671801911253396610845811665058135986328125,
		1.1018277869184245343348038659314624965190887451171875,
		2.27232646480507671782334000454284250736236572265625,
		0.79461119721747242028442315131542272865772247314453125,
		0.82123036225757306549866143541294150054454803466796875
	},
	{
		1.07173918114691790748338462435640394687652587890625,
		1.5946397959245037778686082674539647996425628662109375,
		1.8156834286023630919970628383453004062175750732421875,
		5.252652531207417041514418087899684906005859375,
		1.6512801835071024836310016326024197041988372802734375,
		4.26617154334360382250679322169162333011627197265625,
		2.203655573836849068669607731862924993038177490234375,
		4.5446529296101534356466800090856850147247314453125,
		1.5892223944349448405688463026308454573154449462890625,
		1.6424607245151461309973228708258830010890960693359375
	},
	{
		2.1434783622938358149667692487128078937530517578125,
		3.189279591849007555737216534907929599285125732421875,
		3.631366857204726183994125676690600812435150146484375,
		10.50530506241483408302883617579936981201171875,
		3.302560367014204967262003265204839408397674560546875,
		8.5323430866872076450135864433832466602325439453125,
		4.40731114767369813733921546372584998607635498046875,
		9.089305859220306871293360018171370029449462890625,
		3.178444788869889681137692605261690914630889892578125,
		3.284921449030292261994645741651766002178192138671875
	}
};



template<typename T>
static
__global__ void test_generateValues(
	T* const output,
	const std::size_t count,
	const unsigned long long seed,
	const unsigned long long subsequence,
	const float shape,
	const float scale)
{
	OpenMPCD::CUDA::Random::Generators::Philox4x32_10 rng(seed, subsequence);
	OpenMPCD::CUDA::Random::Distributions::Gamma<T> dist(shape, scale);

	for(std::size_t i = 0; i < count; ++i)
		output[i] = dist(rng);
}
SCENARIO(
	"`OpenMPCD::CUDA::Random::Distributions::Gamma` "
		"returns expected values",
	"[CUDA]")
{
	const std::size_t count = expectedValuesCount;
	static const unsigned long long seed = 1337;
	static const unsigned long long subsequence = 42;

	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	float* d_output_float;
	double* d_output_double;
	dmm.allocateMemory(&d_output_float, count);
	dmm.allocateMemory(&d_output_double, count);

	for(std::size_t p = 0; p < parameterCombinationCount; ++p)
	{
		const float shape = parameterCombinations[p][0];
		const float scale = parameterCombinations[p][1];

		const float* const expected_float = expected_float_array[p];
		const double* const expected_double = expected_double_array[p];


		test_generateValues<<<1, 1>>>(
			d_output_float, count, seed, subsequence, shape, scale);
		test_generateValues<<<1, 1>>>(
			d_output_double, count, seed, subsequence, shape, scale);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;


		float results_float[count];
		double results_double[count];
		dmm.copyElementsFromDeviceToHost(
				d_output_float, results_float, count);
		dmm.copyElementsFromDeviceToHost(
			d_output_double, results_double, count);

		for(std::size_t i = 0; i < count; ++i)
		{
			//results differ slightly for different GPU models
			REQUIRE(results_float[i] == Approx(expected_float[i]));
			REQUIRE(results_double[i] == Approx(expected_double[i]));
		}
	}
}


template<typename T>
static
__global__ void test_checkRange(
	bool* const status,
	const std::size_t count,
	const unsigned long long seed,
	const T shape,
	const T scale)
{
	*status = true;

	OpenMPCD::CUDA::Random::Generators::Philox4x32_10 rng(seed, 0);
	OpenMPCD::CUDA::Random::Distributions::Gamma<T> dist(shape, scale);

	for(std::size_t i = 0; i < count; ++i)
	{
		const T value = dist(rng);
		if(value <= 0)
			*status = false;
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::Random::Distributions::Gamma` returns positive values",
	"[CUDA]")
{
	static const std::size_t count = 1000;

	const unsigned long long seed = std::time(0);

	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);
	bool* d_status_float;
	bool* d_status_double;
	dmm.allocateMemory(&d_status_float, 1);
	dmm.allocateMemory(&d_status_double, 1);

	for(std::size_t p = 0; p < parameterCombinationCount; ++p)
	{
		const float shape = parameterCombinations[p][0];
		const float scale = parameterCombinations[p][1];

		dmm.zeroMemory(d_status_float, 1);
		dmm.zeroMemory(d_status_double, 1);

		test_checkRange<float><<<1, 1>>>(
			d_status_float, count, seed, shape, scale);
		test_checkRange<double><<<1, 1>>>(
			d_status_double, count, seed, shape, scale);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		static const bool true_ = true;
		REQUIRE(
			dmm.elementMemoryEqualOnHostAndDevice(&true_, d_status_float, 1));
		REQUIRE(
			dmm.elementMemoryEqualOnHostAndDevice(&true_, d_status_double, 1));
	}
}