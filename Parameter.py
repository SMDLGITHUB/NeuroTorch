import math
import sys

#Chip Specifications
GlobalBusWidth = 8192   # Global Bus width
NumTilePerChip = 16     # Number of Tiles per Chip
NumPEperTile = 4   # Number of PEs per Tile
NumSubperPE = 8    # Number of SubArrays per PE
SubRow = 256    # Row Size of SubArray
SubCol = 256    # Column SIze of Sub Array
PEBuffer = 'SRAM'   # The type of Cache for PE (There are 'SRAM' ,  'DFF')
TileBuffer = 'SRAM'   # The type of Cache for Tile (There are 'SRAM' ,  'DFF')
RowsPerADC = 8
ColumnsPerADC = 8   # ADC sharing using MUX (For Area saving)
ColumnsPerWeight = 1    # For Onchip Training, Column per weight should be 1 (For Parallel update)
SRAMRow = 128       # Row Size of SRAM Cache
SRAMRowF = 8
SRAMCol = 128       # Column Size of SRAM Cache
SRAMColF = 20
BusToTileT = 'HTree'     # Type of InterConnect to Tile ('HTree') (We now only coded only about HTree)
BusToPET = 'HTree'     # Type of InterConnect to PE ('HTree' ,  'Bus')
BusToSubT = 'Bus'     # Type of InterConnect to SubArray ('HTree' ,  'Bus')
PEAccParr = 32       # Copy of AdderTree for PE to Process Accumulation Parallely. (If 1, only 1 Adder Tree exists)
TileAccParr = 64    # Copy of AdderTree for Tile to Process Accumulation Parallely. (If 1, only 1 Adder Tree exists)
GlobalAccParr = 64   # Copy of AdderTree for Chip to Process Accumulation Parallely. (If 1, only 1 Adder Tree exists)

# Elements which is determined by Network
IMGSIZE = 32
MaxLayerInput = 32 * 32 * 64
BATCH = 1
LearnRate = 0.0005
UpdateMax = 0.0005
EPOCH = 1
NUMCLASS = 10   # Number of Answer classes

# Network property
DATANUM = 100
NumLayer = 9
Kernel = 3
LayerIMG = [32, 32, 16, 16, 8, 8, 8, 4096, 1024, 10]
LayerInChan = [3, 64, 64, 128, 128, 256, 256, 4096, 1024, 10]
LayerType = ['Conv', 'Conv','Conv','Conv','Conv','Conv','Conv','FC','FC', 'END']
IsPoolLayer = [0, 1, 0, 1, 0, 0, 1, 0, 0]
SpeedUP = [64, 4, 4, 4, 2, 2, 2, 1, 16] # For NeuroSim Comparison

# Elements especially related to the quantization
Mode = 'Onchip_Parallel'   # What CIM supports (There are 'Inference_Normal' , 'Onchip_Normal', 'Onchip_Parallel')
Onchip_ParMode = 'LTP_only'         # Use only LTP or LTD or both LTP, LTD :  'LTP_only', 'LTD_only', 'LTPLTD_both'
MUXSize = 1     # MUX_Width Size (1 means Minimum Feature Size)
ADCType = 'SAR'
ADCPrecision = 5    # ADC Precision for CIM Acceleration
ADCMax = 50
DelEn = 1
DelPrecision = 14
DelADCMax = 0.1
ADCCycle = 1e-9     # ADC clock cycle (SAR ADC)
InputEncoding = 1    # How many bits per input cycle
InputPrecesion = 8      # How many bit resolutions per input
Cycle = 1e-9

#Memory Device Specifications
Memory = 'Flash'    # Type of Memory (There are 'Flash' , 'ReRAM' , 'FeFET' 'IDEAL')
if Memory == 'Flash':
    HighVoltage = 9  # The voltage used for a level shifter for Memory Write Operations
    MemHeight = 2
    MemWidth = 4
    ONCurrent = 1e-8
    OnOffRatio = 10
    WriteCurr = 100e-15
    InhibCurr = 0
    FullTimeLTP = 100 * 1e-3
    FUllTimeLTD = 200 * 100e-6
    OnePulseLTP = 1e-3
    OnePulseLTD = 100e-6
    LTP = 0     # LTP Nonlinearity (if linear, write 0)
    LTD = 4.7     # LTD Nonlinearity (if linear, write 0)
    Vari = 0.6    # Device variation
    LEVEL = 200
    ReadTimePerCycle = 5e-9 # Required Readtime per cycle
    VRead = 1
    UpdateLEVELs = min(2 * math.floor(UpdateMax * min(FullTimeLTP, FUllTimeLTD) / (2 * Cycle)),
                       2 ** (InputPrecesion + DelPrecision - math.floor(math.log2(1 / UpdateMax))))


if Memory == 'FeFET': # From "Ferroelectric FET Analog Synapse for Acceleration of Deep Neural Network Training"
    HighVoltage = 3.7  # The voltage used for a level shifter for Memory Write Operations
    MemHeight = 4   # Not Known
    MemWidth = 4   # Not Known
    ONCurrent = 8.9e-8
    OnOffRatio = 45
    WriteCurr = 100e-15
    InhibCurr = 0
    FullTimeLTP = 32 * 75e-9
    FUllTimeLTD = 32 * 75e-9
    OnePulseLTP = 75e-9
    OnePulseLTD = 75e-9
    LTP = -1.75     # LTP Nonlinearity (if linear, write 0)
    LTD = 1.46     # LTD Nonlinearity (if linear, write 0)
    Vari = 0.6    # Device variation
    LEVEL = 32
    ReadTimePerCycle = 5e-9 # Required Readtime per cycle
    VRead = 0.05
    UpdateLEVELs = 1 #min(2 * math.floor(UpdateMax * min(FullTimeLTP, FUllTimeLTD) / (2 * Cycle)),
                       #2 ** (22 - math.floor(math.log2(1 / UpdateMax))))

if Memory == 'ReRAM': # From "Nanoscale Memristor Device as Synapse in Neuromorphic Systems"
    HighVoltage = 3.2  # The voltage used for a level shifter for Memory Write Operations
    MemHeight = 4   # Not Known
    MemWidth = 4    # Not Known
    ONCurrent = 40e-9
    OnOffRatio = 12.5
    WriteCurr = 64e-9
    InhibCurr = 32e-9
    FullTimeLTP = 300e-6 * 100
    FUllTimeLTD = 300e-6 * 100
    OnePulseLTP = 300e-6
    OnePulseLTD = 300e-6
    LTP = 2.4     # LTP Nonlinearity (if linear, write 0)
    LTD = 4.9     # LTD Nonlinearity (if linear, write 0)
    Vari = 0.6    # Device variation
    LEVEL = 100
    ReadTimePerCycle = 5e-9 # Required Readtime per cycle
    VRead = 2
    UpdateLEVELs = min(2 * math.floor(UpdateMax * min(FullTimeLTP, FUllTimeLTD) / (2 * Cycle)),
                       2 ** (22 - math.floor(math.log2(1 / UpdateMax))))

if Memory == 'ReRAM2': # From "A methodology to improve linearity of analog RRAM for neuromorphic computing"
    HighVoltage = 1.6  # The voltage used for a level shifter for Memory Write Operations
    MemHeight = 2   # Not Known
    MemWidth = 4    # Not Known
    ONCurrent = 40e-9
    OnOffRatio = 10
    WriteCurr = 16e-6
    InhibCurr = 8e-6
    FullTimeLTP = 50e-9 * 128
    FUllTimeLTD = 50e-9 * 128
    OnePulseLTP = 50e-9
    OnePulseLTD = 50e-9
    LTP = 0     # LTP Nonlinearity (if linear, write 0)
    LTD = 0.63     # LTD Nonlinearity (if linear, write 0)
    Vari = 0.6    # Device variation
    LEVEL = 128
    ReadTimePerCycle = 5e-9 # Required Readtime per cycle
    VRead = 0.8
    UpdateLEVELs = min(2 * math.floor(UpdateMax * min(FullTimeLTP, FUllTimeLTD) / (2 * Cycle)),
                       2 ** (22 - math.floor(math.log2(1 / UpdateMax))))


if Memory == 'IDEAL': # From "Nanoscale Memristor Device as Synapse in Neuromorphic Systems"
    HighVoltage = 1  # The voltage used for a level shifter for Memory Write Operations
    MemHeight = 2   # Not Known
    MemWidth = 2    # Not Known
    ONCurrent = 10e-9
    OnOffRatio = 10000
    WriteCurr = 100e-15
    InhibCurr = 0
    FullTimeLTP = 50e-6 * 100
    FUllTimeLTD = 50e-6 * 100
    OnePulseLTP = 50e-6
    OnePulseLTD = 50e-6
    LTP = 0     # LTP Nonlinearity (if linear, write 0)
    LTD = 0     # LTD Nonlinearity (if linear, write 0)
    Vari = 0.6    # Device variation
    LEVEL = 1000
    ReadTimePerCycle = 5e-9 # Required Readtime per cycle
    VRead = 0.2
    UpdateLEVELs = min(2 * math.floor(UpdateMax * min(FullTimeLTP, FUllTimeLTD) / (2 * Cycle)),
                       2 ** (22 - math.floor(math.log2(1 / UpdateMax))))

if UpdateLEVELs < 2:
    print('Too low Update Resolution!!')
    #sys.exit()

UPDEL = 0
for i in range(InputPrecesion):
    UPDEL += math.ceil(math.log2(UpdateLEVELs)) - InputPrecesion + i


# Design Constraints
Tech = 45   # Technology Node Chosen (There is 45 nm now (32, 28, 22 are also supported))
if Tech == 45:
    VDD = 1
    EFFR = 1.05     # Effective Resistance multiplier we use
    CONSTEFFR = 1.54    # Effective Resistance multiplier S.Yu use
    WireWidth = 50
    UnitWiRes = 4e-8 / (WireWidth * WireWidth * 1e-18 * 1.9)
    NONCurr = 45 * 1e-9 * 1.27e3    # NMOS On Current with mimimum feature size
    PONCurr = 45 * 1e-9 * 1.08e3    # PMOS On Current with mimimum feature size
    NONCurrS = 45 * 1e-9 * 1.15e3  # NMOS On Current with mimimum feature size (SPICE)
    PONCurrS = 45 * 1e-9 * 0.70e3  # PMOS On Current with mimimum feature size (SPICE)
    JuncCapP = 0.05e-15
    JuncCapN = 0.05e-15
    GateCapP = 0.04e-15
    GateCapN = 0.04e-15
    PNRATIO = 2     # Ratio Between PMOS and NMOS
    RA = 0.18   # Unit Small wire resistance
    MIGRATIONLIM = WireWidth * WireWidth * 1.5e-8

PNGAP = 3   # GAP Between PMOS Diffusion Area and NMOS Diffusion Area
GATEGAP = 3     # GAP Between Field GATEs
GATEOVERLAP = 1     # Gate Length which is overlapped(extended) according to the Design RuleFu
CELLHeight = 28     # Height of Standard Cell of Mask Layout
GAP_BET_GATE_POLY = 3   # Gate poly GAP for Width Calculation
BUSFOLD = 4     # Folded Bus Ratio (4)
MetalPitch = 3  # Metal Pitch

# Physical Characteristics
UnitWiCap = 1e-17/1e-6      # 0.01fF/um

Dataset = 'cifar10'       # Train data ('mnist', 'cifar10', 'cifar100')