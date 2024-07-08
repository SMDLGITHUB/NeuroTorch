import torch
import Parameter
import math

# Module Areas
Total = 0
Chip_width = 0
Chip_height = 0
Tile_width = 0
Tile_height = 0
Tile_area = 0
PE_width = 0
PE_height = 0
PE_area = 0
Subarray_width = 0
Subarray_height = 0
Subarray_area = 0

InterConnection = 0
GlobalInterConnection = 0
TileInterConnection = 0
PEInterConnection = 0

Accumul = 0
GlobalAccumul = 0
TileAccumul = 0
PEAccumul = 0

PEBuffer = 0
TileBuffer = 0
GlobalBufferC = 0
TotalBuffer = 0

#Element Areas
DFFC = 0
ShiftAdderC = 0
MUXC = 0
ADCC = 0
DriverC = 0
LevelShifterC = 0
MemoryArrayC = 0

# Number of Buffer Cores
NumG = 0
NumITBG = 0
NumOTBG = 0
NumIPEBG = 0
NumOPEBG = 0

def Calculate(): # Calculte the area of Each component in Chip
    global Total
    global Chip_height
    global Chip_width
    global TotalBuffer
    global GlobalBufferC
    global InterConnection
    global GlobalInterConnection
    global Accumul
    global GlobalAccumul
    global NumG

    if Parameter.Mode == 'Inference_Normal':
        TileH, TileW = Tile()
        # InterConnect Area
        IntCArea = InterConnect(Parameter.GlobalBusWidth, Parameter.NumTilePerChip, Parameter.BusToTileT, TileH, TileW)

        # Global Accumulator Area
        ChipAccArea = Accumulate(Parameter.NumTilePerChip) * Parameter.GlobalAccParr

        # Global Buffer Area
        GlobalBuffer = SRAMBuffer()
        NumGBuff = math.ceil((Parameter.InputPrecesion+1) * Parameter.MaxLayerInput /(Parameter.SRAMRow * Parameter.SRAMCol))
        GlobalBuffer *= NumGBuff
        NumG = NumGBuff
        # Total Area
        Total = Parameter.NumTilePerChip * (TileH * TileW) + IntCArea + ChipAccArea + GlobalBuffer

        Chip_height = math.sqrt(Total)
        Chip_width = Total / Chip_height

    if Parameter.Mode == 'Onchip_Normal':
        TileH, TileW = Tile()
        # InterConnect Area
        IntCArea = InterConnect(Parameter.GlobalBusWidth, Parameter.NumTilePerChip, Parameter.BusToTileT, TileH, TileW)

        # Global Accumulator Area
        ChipAccArea = Accumulate(Parameter.NumTilePerChip) * Parameter.GlobalAccParr

        # Global Buffer Area
        GlobalBuffer = SRAMBuffer()
        NumGBuff = math.ceil((Parameter.InputPrecesion+1) * Parameter.MaxLayerInput /(Parameter.SRAMRow * Parameter.SRAMCol))
        GlobalBuffer *= NumGBuff

        # Total Area
        Total = Parameter.NumTilePerChip * (TileH * TileW) + IntCArea + ChipAccArea + GlobalBuffer

        Chip_height = math.sqrt(Total)
        Chip_width = Total / Chip_height

    if Parameter.Mode == 'Onchip_Parallel':
        TileH, TileW = Tile()
        # InterConnect Area
        IntCArea = InterConnect(Parameter.GlobalBusWidth, Parameter.NumTilePerChip, Parameter.BusToTileT, TileH, TileW)

        # Global Accumulator Area
        ChipAccArea = Accumulate(Parameter.NumTilePerChip) * Parameter.GlobalAccParr

        # Global Buffer Area
        GlobalBuffer = SRAMBuffer()
        Bit = max((Parameter.InputPrecesion+1), (Parameter.DelPrecision + 1))
        NumGBuff = math.ceil(Bit * Parameter.MaxLayerInput * Parameter.NumLayer / (Parameter.SRAMRow * Parameter.SRAMCol))
        GlobalBuffer *= NumGBuff
        NumG = NumGBuff

        # Total Area
        Total = Parameter.NumTilePerChip * (TileH * TileW) + IntCArea + ChipAccArea + GlobalBuffer

        Chip_height = math.sqrt(Total)
        Chip_width = Total / Chip_height

    GlobalInterConnection += IntCArea
    GlobalAccumul += ChipAccArea
    GlobalBufferC += GlobalBuffer
    TotalBuffer += GlobalBuffer
    InterConnection += IntCArea
    Accumul += ChipAccArea


def Tile(): #Get Unit Area of single Tile
    global Tile_area
    global NumITBG
    global NumOTBG
    global InterConnection
    global TileInterConnection
    global TotalBuffer
    global TileBuffer
    global Accumul
    global TileAccumul

    PEH, PEW = PE()

    # InterConnection Area
    if Parameter.BusToTileT == 'HTree':
        PEBitWidth =  max(Parameter.SRAMRow, Parameter.SubRow)
        PEBitWidth *= Parameter.NumTilePerChip
    if Parameter.BusToPET == 'HTree':
        PEBitWidth =  max(Parameter.SRAMRow, Parameter.SubRow)

    IntCArea =  InterConnect(PEBitWidth, Parameter.NumPEperTile, Parameter.BusToPET, PEH, PEW)

    # Tile Accumulator Area
    TileAccArea = Accumulate(Parameter.NumPEperTile) * Parameter.TileAccParr

    # Tile Input Buffer Area
    if Parameter.TileBuffer == 'SRAM':
        TileBuffArea = SRAMBuffer()
        IMG = Parameter.SRAMRow * Parameter.SRAMCol
        NumITBG = math.ceil((Parameter.InputPrecesion + 1) * Parameter.NumPEperTile * Parameter.SubRow / IMG)
        NumITBBP = math.ceil((Parameter.DelPrecision + 1) * Parameter.NumPEperTile * Parameter.SubRow / IMG)
        if Parameter.Mode == 'Inference_Normal':
            TileBuffArea *= NumITBG
        if Parameter.Mode == 'Onchip_Parallel':
            TileBuffArea *= (NumITBG + NumITBBP)

        # Not Used Yet
    if Parameter.TileBuffer == 'DFF':
        TileBuffArea = DFF()

    # Tile Output Buffer Area
    if Parameter.TileBuffer == 'SRAM':
        TileOutBuffArea = SRAMBuffer()
        IMG = Parameter.SRAMRow * Parameter.SRAMCol
        Bit1 = Parameter.InputPrecesion + 1
        Bit2 = Parameter.DelPrecision + 1
        NumOTB = math.ceil(Bit1 * Parameter.NumPEperTile * Parameter.SubCol / Parameter.ColumnsPerADC / IMG)
        NumOTBG = NumOTB
        NumOTBBP = math.ceil(Bit2 * Parameter.NumPEperTile * Parameter.SubRow / Parameter.RowsPerADC / IMG)
        if Parameter.Mode == 'Inference_Normal':
            TileOutBuffArea *= NumOTBG
        if Parameter.Mode == 'Onchip_Parallel':
            TileOutBuffArea *= (NumOTBG + NumOTBBP)


        # Not Used Yet
    if Parameter.TileBuffer == 'DFF':
        TileBuffArea = DFF()

    # Total Area
    TotalTile = Parameter.NumPEperTile * (PEH * PEW) + IntCArea + TileAccArea + TileBuffArea + TileOutBuffArea

    Tile_area += TotalTile
    NumCopy = Parameter.NumTilePerChip
    TileAccumul += TileAccArea * NumCopy
    Accumul += TileAccArea * NumCopy
    TotalBuffer += (TileBuffArea + TileOutBuffArea) * NumCopy
    TileBuffer += (TileBuffArea + TileOutBuffArea) * NumCopy
    InterConnection += IntCArea * NumCopy
    TileInterConnection += IntCArea * NumCopy

    Tile_height = math.sqrt(TotalTile)
    Tile_width = TotalTile/Tile_height

    return Tile_height, Tile_width

def PE(): #Get Unit Area of single PE
    global PE_height
    global PE_width
    global NumIPEBG
    global NumOPEBG
    global TotalBuffer
    global PEBuffer
    global InterConnection
    global PEInterConnection
    global Accumul
    global PEAccumul

    SubH, SubW, SubU = SubArray()

    # InterConnection Array
    if Parameter.BusToSubT == 'HTree':
        SubBitWidth = Parameter.SubRow
        SubBitWidth *= Parameter.NumSubperPE

    if Parameter.BusToSubT == 'Bus':
        if Parameter.Mode == 'Inference_Normal':
            SubBitWidth = Parameter.SubRow + math.ceil(Parameter.SubCol/Parameter.ColumnsPerADC)
        if Parameter.Mode == 'Onchip_Parallel':
            Case1 = Parameter.SubRow + math.ceil(Parameter.SubCol/Parameter.ColumnsPerADC)
            Case2 = Parameter.SubCol + math.ceil(Parameter.SubRow/Parameter.RowsPerADC)
            SubBitWidth = max(Case1, Case2)

    IntCArea = InterConnect(SubBitWidth, Parameter.NumSubperPE, Parameter.BusToSubT, SubH, SubW)
    # PE Accumulator Area
    PEAccArea = Accumulate(Parameter.NumSubperPE) * Parameter.PEAccParr

    # PE Input Buffer Area
    if Parameter.PEBuffer == 'SRAM':
        PEBuffArea = SRAMBuffer()
        IMG = Parameter.SRAMCol * Parameter.SRAMRow
        NumIPEBG = math.ceil((Parameter.InputPrecesion + 1) * Parameter.NumSubperPE * Parameter.SubRow / IMG)
        NumIPEBG2 = math.ceil((Parameter.InputPrecesion + 1) * Parameter.NumSubperPE * Parameter.SubRow / IMG)
        NumIPEBBP = math.ceil((Parameter.DelPrecision + 1) * Parameter.NumSubperPE * Parameter.SubCol / IMG)
        if Parameter.Mode == 'Inference_Normal':
            PEBuffArea *= NumIPEBG
        if Parameter.Mode == 'Onchip_Parallel':
            PEBuffArea *= (NumIPEBG + NumIPEBBP)

        # Not Used Yet
    if Parameter.PEBuffer == 'DFF':
        PEBuffArea = DFF()

    # PE Output Buffer Area
    if Parameter.PEBuffer == 'SRAM':
        PEOutBuffArea = SRAMBuffer()
        IMG = Parameter.SRAMCol * Parameter.SRAMRow
        RCANUM = Parameter.ADCPrecision + math.ceil(Parameter.InputPrecesion / Parameter.InputEncoding)
        RCANUM += (Parameter.ColumnsPerWeight - 1)
        NumOPEBG = math.ceil(RCANUM * Parameter.NumSubperPE * Parameter.SubCol / Parameter.ColumnsPerADC / IMG)
        RCANUM2 = Parameter.ADCPrecision + math.ceil(Parameter.DelPrecision / Parameter.DelEn)
        # ColumnPerWeight should be 1 in Onchip_Parallel Now

        NumOPEBBP = math.ceil(RCANUM * Parameter.NumSubperPE * Parameter.SubRow / Parameter.RowsPerADC / IMG)

        if Parameter.Mode == 'Inference_Normal':
            PEOutBuffArea *= NumOPEBG
        if Parameter.Mode == 'Onchip_Parallel':
            PEOutBuffArea *= (NumOPEBG + NumOPEBBP)

        # Not Used Yet
    if Parameter.PEBuffer == 'DFF':
        PEOutBuffArea = DFF()


    TotalPE =  Parameter.NumSubperPE * SubU + IntCArea + PEAccArea + PEBuffArea + PEOutBuffArea

    NumCopy = Parameter.NumPEperTile * Parameter.NumTilePerChip
    Accumul += PEAccArea * NumCopy
    PEAccumul += PEAccArea * NumCopy
    InterConnection += IntCArea * NumCopy
    PEInterConnection += IntCArea * NumCopy
    TotalBuffer += (PEBuffArea + PEOutBuffArea) * NumCopy
    PEBuffer += (PEBuffArea + PEOutBuffArea) * NumCopy

    PE_height = math.sqrt(TotalPE)
    PE_width = TotalPE/PE_height
    return PE_height, PE_width

def SubArray(): #Get Unit Area of single SubArray
    global Subarray_width
    global Subarray_height
    global ADCC
    global DriverC
    global MemoryArrayC
    global MUXC
    global ShiftAdderC
    global LevelShifterC

    Used = 0

    if Parameter.Mode == 'Inference_Normal':
        MH, MW = MemoryArray()
        DH, DW = Driver(MH, 0)
        LSH, LSW = Levelshifter(MH, 0)
        MUXH, MUXW = 0, 0
        if Parameter.ColumnsPerADC > 1:
            MUXH, MUXW, DecA = MUX(0, MW)
        ADCH, ADCW = ADC(0, MW)
        ShAdH, ShAdW = ShiftAdd(0, MW)
        Subarray_height = MH + MUXH + ADCH + ShAdH
        Subarray_width = MW + DW + LSW
        Used += MW * Subarray_height
        Used += MH * Subarray_width
        Used += DecA
        Used -= MH * MW
        ADCC += ADCH * MW
        DriverC += MH * DW
        LevelShifterC += MH * LSW
        MUXC += MW * MUXH + DecA
        ShiftAdderC += MW * ShAdH
        MemoryArrayC += MW * MH

    if Parameter.Mode == 'Onchip_Normal':
        # FF Side
        MH, MW = MemoryArray()
        DH, DW = Driver(MH, 0)
        LSH, LSW = Levelshifter(MH, 0)
        if Parameter.ColumnsPerADC > 1:
            MUXH, MUXW = MUX(0, MW)
        ADCH, ADCW = ADC(0, MW)
        ShAdH, ShAdW = ShiftAdd(0, MW)

        #BP Side

        DBPH, DBP = Driver(0, MW)
        LSBPH, LSBPW = Levelshifter(0, MW)

        Subarray_height = 1
        Subarray_width = 1

    if Parameter.Mode == 'Onchip_Parallel':
        MH, MW = MemoryArray()
        DH, DW = Driver(MH, 0)
        LSH, LSW = Levelshifter(MH, 0)
        MUXH, MUXW = 0, 0
        if Parameter.ColumnsPerADC > 1:
            MUXH, MUXW, DecA = MUX(0, MW)
        ADCH, ADCW = ADC(0, MW)
        ShAdH, ShAdW = ShiftAdd(0, MW)

        DHBP, DWBP  = Driver(0, MW)
        LSHBP, LSWBP = Levelshifter(0, MW)
        MUXHBP, MUXWBP = 0, 0
        if Parameter.ColumnsPerADC > 1:
            MUXHBP, MUXWBP, DecABP = MUX(MH, 0)
        ADCHBP, ADCWBP = ADC(MH, 0)
        ShAdHBP, ShAdWBP = ShiftAdd(MH, 0)

        Subarray_height = MH + MUXH + ADCH + ShAdH + DHBP + LSHBP
        Subarray_width = MW + DW + LSW + MUXWBP + ADCWBP + ShAdWBP
        Used += MW * Subarray_height
        Used += MH * Subarray_width
        Used += DecA + DecABP
        Used -= MH * MW
        ADCC += ADCH * MW + MH * ADCWBP
        DriverC += MH * DW + MW * DHBP
        LevelShifterC += MH * LSW + MW * LSHBP
        MUXC += MW * MUXH + DecA + MH * MUXWBP + DecABP
        ShiftAdderC += MW * ShAdH + MH * ShAdWBP
        MemoryArrayC += MW * MH

    return  Subarray_height, Subarray_width, Used

def Driver(Height, Width): #Get Area of Drivers

    if Parameter.HighVoltage > Parameter.VDD:       # Use an AND gate besides Levelshifter for Driver
        MAXTRANWidth = Parameter.CELLHeight - Parameter.PNGAP - (2 * Parameter.GATEOVERLAP) - Parameter.GATEGAP

        # NAND
        Width1 = (2 + Parameter.PNRATIO)
        NUMIN = 2
        NUMFOLD1 = math.ceil(Width1 / MAXTRANWidth)
        W1 = (NUMFOLD1 + 1) * NUMIN * (1+ Parameter.GAP_BET_GATE_POLY) * 1e-6 * Parameter.Tech

        # INV
        Width2 = 3
        NUMFOLD2 = math.ceil(Width2 / MAXTRANWidth)
        W2 = (NUMFOLD2 + 1) * (1+ Parameter.GAP_BET_GATE_POLY) * 1e-6 * Parameter.Tech
        Driver_WidthUn = W1 + W2
        Driver_HeightUn = Parameter.CELLHeight * Parameter.Tech * 1e-6
        A = Driver_HeightUn * Driver_WidthUn

    if Parameter.HighVoltage <= Parameter.VDD:       # In this case, we can adopt WLDriver Suggested by S.Yu
        MINTGHEIGHT = Parameter.CELLHeight * Parameter.Tech * 1e-6
        MAXTRANWidth = Parameter.CELLHeight - Parameter.PNGAP - (2 * Parameter.GATEOVERLAP) - Parameter.GATEGAP
        WidthTG = (1 + Parameter.PNRATIO)
        NUMFOLDTG = math.ceil(WidthTG / MAXTRANWidth)
        WTG1 = (NUMFOLDTG + 1) * (1 + Parameter.GAP_BET_GATE_POLY) * 1e-6 * Parameter.Tech
        WTG = WTG1 * 4
        A = MINTGHEIGHT * WTG

    D = DFF()
    Driver = A + D

    # Calculating Total Driver Area
    if Width == 0:
        Driver_Height = Height
        Driver_Width = (A * Parameter.SubRow) / Driver_Height

    if Height == 0:
        Driver_Width = Width
        Driver_Height = (A * Parameter.SubCol) / Driver_Width

    return Driver_Height, Driver_Width


def PWM(): #Get Unit Area of single PWM
    return 0 # This Part will be later used.

def Levelshifter(Height, Width): #Get Area of Levelshifters

    # Refer to "22nm SOC Platform Technology Featuring~","High Voltage Devices and Circuits in Standard CMOS Technology"
    if Parameter.HighVoltage > 3.3:
        Min_IOLeng = 270e-6 * math.sqrt(Parameter.HighVoltage / 3.3)    #Considering Punch Through
    if Parameter.HighVoltage <= 3.3:
        Min_IOLeng = 270e-6
    if Parameter.HighVoltage <= Parameter.VDD:
        return 0, 0

    # Calculating Unit LevelShifter Area
    MAXTRANWidth = Parameter.CELLHeight * 2.5 - Parameter.PNGAP - (2 * Parameter.GATEOVERLAP) - Parameter.GATEGAP
    Width1 = (30 + 40) * 2
    NUMFOLD1 = math.ceil(Width1/MAXTRANWidth)
    Width2 = (64 + 20) * 2
    NUMFOLD2 = math.ceil(Width2 / MAXTRANWidth)
    Width3 = (64 + 20) * 2
    NUMFOLD3 = math.ceil(Width3 / MAXTRANWidth)
    Width4 = (128 + 164) * 2
    NUMFOLD4 = math.ceil(Width4 / MAXTRANWidth)

    NUMFOLD = NUMFOLD1 + NUMFOLD2 + NUMFOLD3 + NUMFOLD4

    LevelShifter_HeightUn = Parameter.CELLHeight * Parameter.Tech * 1e-6 * 2.5
    LevelShifter_WidthUn = (NUMFOLD + 1*4) * (Min_IOLeng + (Parameter.GAP_BET_GATE_POLY * 1e-6 * Parameter.Tech))
    A = LevelShifter_HeightUn * LevelShifter_WidthUn

    LevelShifter = A

    # Calculate Total Levelshifters Area in SubArray
    if Width == 0:
        LevelShifter_Height = Height
        LevelShifter_Width =  (A * Parameter.SubRow)/LevelShifter_Height
    if Height == 0:
        LevelShifter_Width = Width
        LevelShifter_Height =  (A * Parameter.SubCol)/LevelShifter_Width

    return LevelShifter_Height, LevelShifter_Width

def MemoryArray(): # Get Unit Area of single Memory Array
    global MemoryArray

    MemoryArray_Height = Parameter. Tech * Parameter.MemHeight * 1e-6 * Parameter.SubRow
    MemoryArray_Width = Parameter. Tech * Parameter.MemWidth * 1e-6 * Parameter.SubCol
    MemoryArray = MemoryArray_Height * MemoryArray_Width
    return MemoryArray_Height, MemoryArray_Width

def MUX(Height, Width): # Get Unit Area of MUXs

    if Parameter.ColumnsPerADC == 1:
        return 0, 0

    # Calculating Unit MUX Area
    MUX_HeightUn = (1+Parameter.PNRATIO) * Parameter.MUXSize+Parameter.PNGAP
    MUX_HeightUn += (2*Parameter.GATEOVERLAP + Parameter.GATEGAP)
    MUX_HeightUn *= Parameter.Tech * 1e-6
    MUX_WidthUn = (1+Parameter.GAP_BET_GATE_POLY) * 2 * Parameter.Tech * 1e-6
    A = MUX_HeightUn * MUX_WidthUn
    NumTG = Parameter.SubCol
    Area = 0
    for i in range(math.ceil(math.log2(Parameter.ColumnsPerADC))):
        Area += NumTG * A
        NumTG /= 2

    MUX = Area
    # Calculate MUX Decoder Area
    # NumPreDec = math.floor(math.ceil(math.log2(Parameter.SRAMRow))/2) # Get Number of 2:4 Predecoders

    NumInNOR = math.ceil(math.log2(Parameter.ColumnsPerADC) / 2)  # Num of Input (NOR GATE)

    # INV for PreDecoder
    HeightUn = Parameter.CELLHeight * Parameter.Tech * 1e-6
    MAXTRANWidth = Parameter.CELLHeight - Parameter.PNGAP - (2 * Parameter.GATEOVERLAP) - Parameter.GATEGAP
    WidthINV = (8 + 8 * Parameter.PNRATIO)
    NUMFOLDINV = math.ceil(WidthINV / MAXTRANWidth)
    WINV = (NUMFOLDINV + 1) * (1 + Parameter.GAP_BET_GATE_POLY) * 1e-6 * Parameter.Tech
    INVA = HeightUn * WINV
    NUMINV = math.ceil(math.log2(Parameter.ColumnsPerADC) / 2.0)

    # NAND2 for PreDecoder
    WidthNAND2 = (8 * Parameter.PNRATIO + 2 * 8)
    NUMINNAND = 2
    NUMFOLDNAND2 = math.ceil(WidthNAND2 / MAXTRANWidth)
    WNAND2 = (NUMFOLDNAND2 + 1) * NUMINNAND * (1 + Parameter.GAP_BET_GATE_POLY) * 1e-6 * Parameter.Tech
    NAND2A = HeightUn * WNAND2
    NUMNAND = 4 * math.floor(math.log2(Parameter.ColumnsPerADC) / 2.0)

    # NOR for RowDecoder
    WidthNOR = (Parameter.PNRATIO * NumInNOR * 8 + 1 * 8)
    NUMFOLDNOR = math.ceil(WidthNOR / MAXTRANWidth)
    WNOR = (NUMFOLDNOR + 1) * NumInNOR * (1 + Parameter.GAP_BET_GATE_POLY) * 1e-6 * Parameter.Tech
    NORA = HeightUn * WNOR
    NUMNOR = Parameter.ColumnsPerADC

    # Metal Connection Considered
    NumMetal = NUMNAND + (math.floor(math.log2(Parameter.ColumnsPerADC)) % 2) * 2
    MetalA = (HeightUn * NumMetal * Parameter.MetalPitch * Parameter.Tech * 1e-6)

    DecA = INVA * NUMINV + NAND2A * NUMNAND + NORA * NUMNOR + MetalA

    # Calculating Total MUX Area
    if Width == 0:
        MUX_Height = Height
        MUX_Width  = (MUX * Parameter.SubRow)/Parameter.ColumnsPerADC/ MUX_Height

    if Height == 0:
        MUX_Width = Width
        MUX_Height = (MUX * Parameter.SubCol)/Parameter.ColumnsPerADC/ MUX_Width

    MUX += DecA

    return MUX_Height, MUX_Width, DecA

def ADC(Height, Width): # Get Area of ADCs

    #Calculating Unit ADC Area
    if Parameter.ADCType == 'SAR':  # Here we adopted Formula From NeuroSim(S. Yu) directly
        HNMOS = Parameter.CELLHeight * Parameter.Tech * 1e-6
        WNMOS = (1 + Parameter.GAP_BET_GATE_POLY) * 2 * Parameter.Tech * 1e-6
        HPMOS = Parameter.CELLHeight * Parameter.Tech * 1e-6
        WPMOS = (1 + Parameter.GAP_BET_GATE_POLY) * 2 * Parameter.Tech * 1e-6
        A = (HNMOS * WNMOS) * (269 + (Parameter.ADCPrecision - 1) * 109)
        A += (HPMOS * WPMOS) * (209 + (Parameter.ADCPrecision-1)*73)

    if Parameter.ADCType == 'Flash':    # Here we use Comparators and Thermometer to Binary Encoder
        A = 0 # Flash will be later Updated

    ADC = A
    # Later our work will be Added

    # Calculating Total ADC Area
    if Width == 0:
        ADC_Height = Height
        ADC_Width  = (A * Parameter.SubRow / Parameter.ColumnsPerADC)/ ADC_Height

    if Height == 0:
        ADC_Width = Width
        ADC_Height = (A * Parameter.SubCol / Parameter.ColumnsPerADC)/ ADC_Width

    return ADC_Height, ADC_Width

def ShiftAdd(Height, Width): # Get Area of single ShiftAdders
    #Get Number of Units for DFF, Full Adder
    numDFF = (Parameter.ADCPrecision + Parameter.InputEncoding)
    numAdder = Parameter.ADCPrecision

    # Get Unit Shift Adder Area

    # Calculate an Unit DFF Area
    MAXTRANWidth = Parameter.CELLHeight - Parameter.PNGAP - (2 * Parameter.GATEOVERLAP) - Parameter.GATEGAP

    # NAND
    Width1 = (2 + Parameter.PNRATIO)
    NUMIN = 2
    NUMFOLD1 = math.ceil(Width1 / MAXTRANWidth)
    W1 = (NUMFOLD1 + 1) * NUMIN * (1 + Parameter.GAP_BET_GATE_POLY) * 1e-6 * Parameter.Tech

    # INV
    Width2 = 3
    NUMFOLD2 = math.ceil(Width2 / MAXTRANWidth)
    W2 = (NUMFOLD2 + 1) * (1 + Parameter.GAP_BET_GATE_POLY) * 1e-6 * Parameter.Tech
    DFF_Width = 8 * W1 + W2
    DFF_Height = Parameter.CELLHeight * Parameter.Tech * 1e-6
    DFF_Width *= numDFF
    A = DFF_Height * DFF_Width

    # Adder
    Adder_Width = W1 * 9 * numAdder
    A += DFF_Height * Adder_Width

    if Height == 0:
        A *= Parameter.SubCol / Parameter.ColumnsPerADC

    if Width == 0:
        A *= Parameter.SubRow / Parameter.ColumnsPerADC

    # Calculating Total ShiftAdder Area
    if Width == 0:
        ShA_Height = Height
        ShA_Width  = A/ ShA_Height

    if Height == 0:
        ShA_Width = Width
        ShA_Height = A / ShA_Width

    return ShA_Height, ShA_Width

def Accumulate(NumtoAcc):  # Get Area of Accumulator (Adder Tree)
    # Single NAND2 Area
    Height = Parameter.CELLHeight * Parameter.Tech * 1e-6
    MAXTRANWidth = Parameter.CELLHeight - Parameter.PNGAP - (2 * Parameter.GATEOVERLAP) - Parameter.GATEGAP
    Width1 = (2 + Parameter.PNRATIO)
    NUMIN = 2
    NUMFOLD1 = math.ceil(Width1 / MAXTRANWidth)
    W1 = (NUMFOLD1 + 1) * NUMIN * (1 + Parameter.GAP_BET_GATE_POLY) * 1e-6 * Parameter.Tech

    # Single FullAdder Area
    FW = 9 * W1
    FAArea = FW * Height

    # Single Adder Tree Area
    AdderTreeArea = 0
    RCANUM = Parameter.ADCPrecision + math.ceil(Parameter.InputPrecesion / Parameter.InputEncoding)
    RCANUM += (Parameter.ColumnsPerADC-1)
    for i in range(0, int(math.floor(math.log2(NumtoAcc)))):
        AdderTreeArea += FAArea * (RCANUM + i) * math.ceil(NumtoAcc/(2**i))

    return AdderTreeArea

def SRAMBuffer(): # Get Unit Area of single SRAM Cache
    # Calculate Cell Area
    SRAMH = Parameter.SRAMRow * Parameter.SRAMRowF * Parameter.Tech * 1e-6
    SRAMW = Parameter.SRAMCol * Parameter.SRAMColF * Parameter.Tech * 1e-6
    CellA = SRAMH * SRAMW
    Height = Parameter.CELLHeight * Parameter.Tech * 1e-6

    # Calculate Decoder Area
    # NumPreDec = math.floor(math.ceil(math.log2(Parameter.SRAMRow))/2) # Get Number of 2:4 Predecoders

    NumInNOR = math.ceil(math.log2(Parameter.SRAMRow)/2)   # Num of Input (NOR GATE)

        # INV for PreDecoder
    MAXTRANWidth = Parameter.CELLHeight - Parameter.PNGAP - (2 * Parameter.GATEOVERLAP) - Parameter.GATEGAP
    WidthINV = (1 + 1 * Parameter.PNRATIO)
    NUMFOLDINV = math.ceil(WidthINV / MAXTRANWidth)
    WINV = (NUMFOLDINV + 1) * (1 + Parameter.GAP_BET_GATE_POLY) * 1e-6 * Parameter.Tech
    INVA = Height * WINV
    NUMINV = math.ceil(math.log2(Parameter.SRAMRow)/2.0)

        # NAND2 for PreDecoder
    WidthNAND2 = (Parameter.PNRATIO + 2)
    NUMINNAND = 2
    NUMFOLDNAND2 = math.ceil(WidthNAND2 / MAXTRANWidth)
    WNAND2 = (NUMFOLDNAND2 + 1) * NUMINNAND * (1 + Parameter.GAP_BET_GATE_POLY) * 1e-6 * Parameter.Tech
    NAND2A = Height * WNAND2
    NUMNAND = 4 * math.floor(math.log2(Parameter.SRAMRow)/2.0)

        # NOR for RowDecoder
    WidthNOR = (Parameter.PNRATIO * NumInNOR + 1)
    NUMFOLDNOR = math.ceil(WidthNOR / MAXTRANWidth)
    WNOR = (NUMFOLDNOR + 1) * NumInNOR * (1 + Parameter.GAP_BET_GATE_POLY) * 1e-6 * Parameter.Tech
    NORA = Height * WNOR
    NUMNOR = Parameter.SRAMRow

        # Metal Connection Considered
    NumMetal = NUMNAND + (math.floor(math.log2(Parameter.SRAMRow)) % 2) * 2
    MetalA = (SRAMH * NumMetal * Parameter.MetalPitch * Parameter.Tech * 1e-6)

    DecA = INVA * NUMINV + NAND2A * NUMNAND + NORA * NUMNOR + MetalA

    # Calculate Precharger Area
    WidthPrePMOS = 6
    WidthEqNMOS = 1
    NUMFOLDPre = math.ceil(WidthPrePMOS / MAXTRANWidth)
    NUMFOLDEq = math.ceil(WidthEqNMOS / MAXTRANWidth)
    UnitPreW = max((NUMFOLDPre+1), (NUMFOLDEq+1)) * Parameter.Tech * 1e-6
    PreCH = Height * 3
    PreCW = UnitPreW * Parameter.SRAMCol
    PreCA = PreCH * PreCW

    # Calculate Write Driver Area
    WriteDWidth = (1 + Parameter.PNRATIO)
    NUMFOLDWriteD = math.ceil(WriteDWidth / MAXTRANWidth)
    UnitWriteDW = (NUMFOLDWriteD + 1) * Parameter.Tech * 1e-6
    WriteDH = Height * 3
    WriteDW = UnitWriteDW * Parameter.SRAMCol
    WriteDA = WriteDH * WriteDW

    # Calculate SenseAmp Area (In NeuroSim, SenseAmp is not considered (Why?))
    SenseWidth = (1 + Parameter.PNRATIO)
    NUMFOLDSense = math.ceil(SenseWidth / MAXTRANWidth)
    UnitSenseW = (NUMFOLDSense + 1) * Parameter.Tech * 1e-6
    SenseH = Height * 3
    SenseW = UnitSenseW * Parameter.SRAMCol
    SenseA = SenseH * SenseW

    A = CellA + DecA + PreCA + WriteDA + SenseA

    return A

def DFF(): # Get Area of DFFs
    # Calculate an Unit DFF Area
    MAXTRANWidth = Parameter.CELLHeight - Parameter.PNGAP - (2 * Parameter.GATEOVERLAP) - Parameter.GATEGAP

    # NAND
    Width1 = (2 + Parameter.PNRATIO)
    NUMIN = 2
    NUMFOLD1 = math.ceil(Width1 / MAXTRANWidth)
    W1 = (NUMFOLD1 + 1) * NUMIN * (1 + Parameter.GAP_BET_GATE_POLY) * 1e-6 * Parameter.Tech

    # INV
    Width2 = 3
    NUMFOLD2 = math.ceil(Width2 / MAXTRANWidth)
    W2 = (NUMFOLD2 + 1) * (1 + Parameter.GAP_BET_GATE_POLY) * 1e-6 * Parameter.Tech
    DFF_Width = 8 * W1 + W2
    DFF_Height = Parameter.CELLHeight * Parameter.Tech * 1e-6
    A = DFF_Height * DFF_Width
    print(DFF_Height * (3*W1 + W2))

    return A

def InterConnect(BitWidth, NumElements, Type, Height, Width): # Get Area of InterConnects
    if Type == 'HTree':
        Area = 0
        # Minimum Inverter CapSize
        if Parameter.Tech == 45:
            MININVCAP = 0.044e-15 + 0.085e-15                # Only Gate Cap (measured with SPICE)
            MININVCAPTO = MININVCAP + 0.049e-15 + 0.11e-15      # Capacitance of (Gate + Drain)

        #Inter Connect Repeater Size and Minimum Distance for Repeater
        ResonRep = (Parameter.CONSTEFFR*Parameter.VDD/Parameter.NONCurr)
        ResonRep += (Parameter.CONSTEFFR * Parameter.VDD / (Parameter.PNRATIO * Parameter.PONCurr))
        REPSIZE = math.floor(math.sqrt(ResonRep * Parameter.UnitWiCap)/(MININVCAP * Parameter.UnitWiRes))
        MINDIST = math.sqrt(2 * ResonRep *MININVCAPTO / (Parameter.UnitWiCap * Parameter.UnitWiRes)) * 1e3

        WIDTHINV = max(1, REPSIZE) * (1 + Parameter.PNRATIO) * 1e-6 * Parameter.Tech
        MAXTRANWidth = Parameter.CELLHeight - Parameter.PNGAP - (2 * Parameter.GATEOVERLAP) - Parameter.GATEGAP
        NUMFOLD2 = math.ceil(WIDTHINV / MAXTRANWidth)
        REPWIDTH = (NUMFOLD2 + 1) * (1 + Parameter.GAP_BET_GATE_POLY) * 1e-6 * Parameter.Tech

        numStage = math.ceil(math.log2(NumElements))
        WireLenH = Height * math.pow(2, (numStage)/2)
        WireLenW = Width * math.pow(2, (numStage)/2)

        # STARTs From the Height
        numREP = 1e-3 * WireLenH / MINDIST
        if numREP >= 1:
            GAP = REPWIDTH
        if numREP < 1:
            GAP = Parameter.WireWidth * 1e-6
        Area += BitWidth * GAP * WireLenH / (2 * Parameter.BUSFOLD)

        # Repeats with Stages
        for i in range(0, int((numStage)/2)):
            WireLenW /= 2
            numREP = 1e-3 * WireLenW / MINDIST
            if numREP >= 1:
                GAP = REPWIDTH
            if numREP < 1:
                GAP = Parameter.WireWidth * 1e-6
            Area += (BitWidth/2) * GAP * WireLenW / Parameter.BUSFOLD

            WireLenH /=2
            numREP = 1e-3 * WireLenH / MINDIST
            if numREP >= 1:
                GAP = REPWIDTH
            if numREP < 1:
                GAP = Parameter.WireWidth * 1e-6
            Area += (BitWidth/2) * GAP * WireLenH / Parameter.BUSFOLD

    if Type == 'Bus':
        # Minimum Inverter CapSize
        if Parameter.Tech == 45:
            MININVCAP = 0.044e-15 + 0.085e-15                # Only Gate Cap (measured with SPICE)
            MININVCAPTO = MININVCAP + 0.049e-15 + 0.11e-15      # Capacitance of (Gate + Drain)

        #Inter Connect Repeater Size and Minimum Distance for Repeater
        ResonRep = (Parameter.CONSTEFFR*Parameter.VDD/Parameter.NONCurr)
        ResonRep += (Parameter.CONSTEFFR * Parameter.VDD / (Parameter.PNRATIO * Parameter.PONCurr))
        REPSIZE = math.floor(math.sqrt(ResonRep * Parameter.UnitWiCap)/(MININVCAP * Parameter.UnitWiRes))
        MINDIST = math.sqrt(2 * ResonRep * MININVCAPTO / (Parameter.UnitWiCap * Parameter.UnitWiRes)) * 1e3

        WIDTHINV = max(1, REPSIZE) * (1 + Parameter.PNRATIO) * 1e-6 * Parameter.Tech
        MAXTRANWidth = Parameter.CELLHeight - Parameter.PNGAP - (2 * Parameter.GATEOVERLAP) - Parameter.GATEGAP
        NUMFOLD2 = math.ceil(WIDTHINV / MAXTRANWidth)
        REPWIDTH = (NUMFOLD2 + 1) * (1 + Parameter.GAP_BET_GATE_POLY) * 1e-6 * Parameter.Tech

        WireLength = math.floor(math.sqrt(NumElements)) * min(Height, Width)
        numREP = 1e-3 * WireLength / MINDIST

        if numREP >= 1:
            GAP = REPWIDTH
        if numREP < 1:
            GAP = Parameter.WireWidth * 1e-6

        Area = BitWidth * GAP * WireLength / Parameter.BUSFOLD

    return Area


def Print(): # Print Areas
    global Total
    global Chip_width
    global Chip_height
    global Tile_area
    global PE_area
    global Subarray_area
    #InterConnetions
    global InterConnection
    global GlobalInterConnection
    global TileInterConnection
    global PEInterConnection

    global DFFC
    global ShiftAdderC
    global MUXC
    global ADCC
    global DriverC
    global LevelShifterC

    # Buffers
    global GlobalBuffer
    global TileBuffer
    global PEBuffer
    global TotalBuffer

    #Accumulators
    global Accumul
    global GlobalAccumul
    global TileAccumul
    global PEAccumul

    global MemoryArrayC

    NUMCOPY = Parameter.NumSubperPE * Parameter.NumTilePerChip * Parameter.NumPEperTile
    print('-------------------------Area---------------------------')
    print('Total Chip Area is {:.9f} mm2'.format(Chip_width * Chip_height))
    print('Chip Height is {:.9f} mm'.format(Chip_height))
    print('Chip Width is {:.9f} mm'.format(Chip_width))
    print('Total InterConnection Area is {:.9f} mm2'.format(InterConnection))
    print('Total Buffer Area is {:.9f} mm2'.format(TotalBuffer))
    print('Total Accumulator Area is {:.9f} mm2'.format(Accumul))
    print('Total Tile Area is {:.9f} mm2'.format(Tile_area * Parameter.NumTilePerChip))
    print('Single Tile Area is {:.9f} mm2'.format(Tile_area))
    print('Total PE Area is {:.9f} mm2'.format(PE_area * Parameter.NumPEperTile * Parameter.NumTilePerChip))
    print('Single PE Area is {:.9f} mm2'.format(PE_area))
    print('Total SubArray Area is {:.9f} mm2'.format(Subarray_area * NUMCOPY))
    print('Single SubArray Area is {:.9f} mm2'.format(Subarray_area))
    print('Total Global Buffer Area is {:.9f} mm2'.format(GlobalBufferC))
    print('Total Tile Buffer Area is {:.9f} mm2'.format(TileBuffer))
    print('Total PE Buffer Area is {:.9f} mm2'.format(PEBuffer))
    print('Total Global Accumulator Area is {:.9f} mm2'.format(GlobalAccumul))
    print('Total Tile Accumulator Area is {:.9f} mm2'.format(TileAccumul))
    print('Total PE Accumulator Area is {:.9f} mm2'.format(PEAccumul))
    print('Total Global InterConnection Area is {:.9f} mm2'.format(GlobalInterConnection))
    print('Total Tile InterConnection Area is {:.9f} mm2'.format(TileInterConnection))
    print('Total PE InterConnection Area is {:.9f} mm2'.format(PEInterConnection))
    print('Total Memory Array Area is {:.9f} mm2'.format(MemoryArrayC * NUMCOPY))
    print('Total Driver Area is {:.9f} mm2'.format(DriverC * NUMCOPY))
    print('Total Levelshifter Area is {:.9f} mm2'.format(LevelShifterC * NUMCOPY))
    print('Total MUX, Decoder Area is {:.9f} mm2'.format(MUXC * NUMCOPY))
    print('Total ShiftAdder Area is {:.9f} mm2'.format(ShiftAdderC * NUMCOPY))
    print('Total ADC Area is {:.9f} mm2'.format(ADCC * NUMCOPY))
    print('---------------------------------------------------------')