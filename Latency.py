import torch
import Parameter
import math
import Area
#import Accuracy

# Module Latency
Total = 0
Chip_Latency = 0
Tile_Latency = 0
PE_Latency = 0
Subarray_Latency = 0

InterConnection_Latency = 0
BufferLatency = 0
AccumulLatency = 0

GlobalInterConnection_Latency = 0
GlobalBufferLatency = 0
GlobalAccumulLatency = 0

TileInterConnection_Latency = 0
TileBufferLatency = 0
TileAccumulLatency = 0

PEInterConnection_Latency = 0
PEBufferLatency = 0
PEAccumulLatency = 0


#Element Latency
ShiftAdderLatency = 0
MUXLatency = 0
ADCLatency = 0
DriverLatency = 0
LevelShifterLatency = 0
MemoryArrayLatency = 0
UpdateLatency = 0

def horowitz(R, CapLoad, rampinput):
    RC = R * CapLoad
    Latency = 2 * RC * math.sqrt(math.log10(0.5) * math.log10(0.5) + 2 * 0.5 * 0.5 / (RC * rampinput))
    return Latency

def Calculate():
    global Total
    global Chip_Latency
    global GlobalBufferLatency
    global GlobalAccumulLatency
    global GlobalInterConnection_Latency
    global InterConnection_Latency
    global AccumulLatency
    global BufferLatency

    TileL = Tile()
    IntCL = InterConnect(Parameter.NumTilePerChip, Parameter.BusToTileT, Area.Tile_height, Area.Tile_width)
    RCANUM = Parameter.ADCPrecision + math.ceil(Parameter.InputPrecesion / Parameter.InputEncoding)
    RCANUM += (Parameter.ColumnsPerWeight-1)
    ChipAccL = Accumulate(Parameter.NumTilePerChip, RCANUM, 0)
    GlobalBuffer = SRAMBuffer(0)

    IntCtotL = 0
    ChipAcctotL = 0
    GlobalBuffertotL = 0
    PoolL = 0
    Total = 0

    if Parameter.Mode == 'Inference_Normal':
        for i in range(Parameter.NumLayer):
            if Parameter.LayerType[i] == 'Conv':
                if Parameter.LayerInChan[i] * Parameter.Kernel * Parameter.Kernel <= Parameter.SubRow:
                    # Number of Subarrays
                    NumT = math.ceil(Parameter.ColumnsPerWeight * Parameter.LayerInChan[i+1] * 2 / Parameter.SubCol)
                    # Number of PE Elements
                    NumT = math.ceil(1.0 * NumT/Parameter.NumSubperPE)
                    # Number of Tile Elements
                    NumT = math.ceil(1.0 * NumT/Parameter.NumPEperTile)

                if Parameter.LayerInChan[i] * Parameter.Kernel * Parameter.Kernel > Parameter.SubRow:
                    # Number of Subarrays
                    if Parameter.LayerType[i + 1] == 'Conv':
                        Parr = min(Parameter.SubRow / Parameter.LayerInChan[i],
                                   Parameter.SubCol / Parameter.LayerInChan[i + 1])
                    if Parameter.LayerType[i + 1] == 'FC':
                        Parr = min(Parameter.SubRow / Parameter.LayerInChan[i],
                                   Parameter.SubCol / Parameter.LayerInChan[i])
                    NumT = (Parameter.Kernel ** 2) * 2 * Parameter.ColumnsPerWeight
                    NumT = math.ceil(NumT/Parr)
                    # Number of PE Elements
                    NumT = math.ceil(1.0 * NumT / Parameter.NumSubperPE)
                    # Number of Tile Elements
                    NumT = math.ceil(1.0 * NumT / Parameter.NumPEperTile)

                # Global HTree Activation
                NumReadIC = (Parameter.LayerIMG[i]**2) * (Parameter.Kernel**2) * Parameter.InputPrecesion
                NumReadIC *= Parameter.LayerInChan[i] / (Parameter.GlobalBusWidth * NumT / Parameter.NumTilePerChip)
                if Parameter.LayerType[i+1] == 'Conv':
                    Next = (Parameter.LayerIMG[i+1]**2) * (Parameter.Kernel**2) * Parameter.InputPrecesion
                    Next *= Parameter.LayerInChan[i+1] / (Parameter.GlobalBusWidth * NumT / Parameter.NumTilePerChip)
                    NumReadIC += Next
                if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                    Next = Parameter.LayerInChan[i+1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))
                    Next /=  (Parameter.GlobalBusWidth * NumT / Parameter.NumTilePerChip)
                    NumReadIC += Next

                # Global Accumulator Activation
                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = (Parameter.LayerIMG[i] ** 2) * Parameter.LayerInChan[i+1] / Parameter.GlobalAccParr

                # Global Buffer Activation
                NumOpBuffer = (Parameter.LayerIMG[i] ** 2) * Parameter.InputPrecesion * (Parameter.Kernel ** 2)
                NumOpBuffer *= Parameter.LayerInChan[i] / (Parameter.SRAMCol * Area.NumG)

                if Parameter.IsPoolLayer[i] == 1:
                    NumReadPool = ((Parameter.LayerIMG[i] / 2) ** 2) * Parameter.LayerInChan[i+1]
                    NumReadPool /= Parameter.GlobalAccParr

                if Parameter.IsPoolLayer[i] == 0:
                    NumReadPool = 0

            if Parameter.LayerType[i] == 'FC':
                # Number of Subarrays
                NumT = math.ceil(Parameter.LayerInChan[i] / Parameter.SubRow) * 2 * Parameter.ColumnsPerWeight
                NumT *= math.ceil(Parameter.LayerInChan[i + 1] / Parameter.SubCol)
                # Number of PE Elements
                NumT = math.ceil(1.0 * NumT / Parameter.NumSubperPE)
                # Number of Tile Elements
                NumT = math.ceil(1.0 * NumT / Parameter.NumPEperTile)

                NumReadIC = Parameter.LayerInChan[i] * Parameter.InputPrecesion
                NumReadIC += Parameter.LayerInChan[i + 1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))
                NumReadIC /= (Parameter.GlobalBusWidth * NumT / Parameter.NumTilePerChip)

                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = Parameter.LayerInChan[i+1] / Parameter.GlobalAccParr

                NumOpBuffer = Parameter.InputPrecesion * Parameter.LayerInChan[i] / (Parameter.SRAMCol * Area.NumG)

            if Parameter.LayerType[i + 1] == 'END':
                NumOpBuffer += (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) \
                               * Parameter.LayerInChan[i+1] / Parameter.SRAMCol
            IntCtotL += NumReadIC * IntCL
            ChipAcctotL += ChipAccL * NumReadAcc
            GlobalBuffertotL += GlobalBuffer * NumOpBuffer
            PoolL += ChipAccL * NumReadPool

        Total += (IntCtotL + ChipAcctotL + GlobalBuffertotL + PoolL) * Parameter.DATANUM

        Total += TileL
    if Parameter.Mode == 'Onchip_Normal':
        for i in range(Parameter.NumLayer):
            NumReadIC = 1
            NumReadAcc = 1
            NumOpBuffer = 1

    if Parameter.Mode == 'Onchip_Parallel':
        for i in range(Parameter.NumLayer):
            if Parameter.LayerType[i] == 'Conv':
                if Parameter.LayerInChan[i] * Parameter.Kernel * Parameter.Kernel <= Parameter.SubRow:
                    # Number of Subarrays
                    NumT = math.ceil(Parameter.ColumnsPerWeight * Parameter.LayerInChan[i+1] * 2 / Parameter.SubCol)
                    # Number of PE Elements
                    NumT = math.ceil(1.0 * NumT/Parameter.NumSubperPE)
                    # Number of Tile Elements
                    NumT = math.ceil(1.0 * NumT/Parameter.NumPEperTile)

                if Parameter.LayerInChan[i] * Parameter.Kernel * Parameter.Kernel > Parameter.SubRow:
                    # Number of Subarrays
                    if Parameter.LayerType[i + 1] == 'Conv':
                        Parr = min(Parameter.SubRow / Parameter.LayerInChan[i],
                                   Parameter.SubCol / Parameter.LayerInChan[i + 1])
                    if Parameter.LayerType[i + 1] == 'FC':
                        Parr = min(Parameter.SubRow / Parameter.LayerInChan[i],
                                   Parameter.SubCol / Parameter.LayerInChan[i])
                    NumT = (Parameter.Kernel ** 2) * 2 * Parameter.ColumnsPerWeight
                    NumT = math.ceil(NumT/Parr)
                    # Number of PE Elements
                    NumT = math.ceil(1.0 * NumT / Parameter.NumSubperPE)
                    # Number of Tile Elements
                    NumT = math.ceil(1.0 * NumT / Parameter.NumPEperTile)

                # Global HTree Activation
                NumReadIC = (Parameter.LayerIMG[i]**2) * (Parameter.Kernel**2) * Parameter.InputPrecesion
                NumReadIC *= Parameter.LayerInChan[i] / (Parameter.GlobalBusWidth * NumT / Parameter.NumTilePerChip)
                if Parameter.LayerType[i+1] == 'Conv':
                    Next = (Parameter.LayerIMG[i+1]**2) * (Parameter.Kernel**2) * Parameter.InputPrecesion
                    Next *= Parameter.LayerInChan[i+1] / (Parameter.GlobalBusWidth * NumT / Parameter.NumTilePerChip)
                    NumReadIC += Next
                if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                    Next = Parameter.LayerInChan[i+1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))
                    Next /=  (Parameter.GlobalBusWidth * NumT / Parameter.NumTilePerChip)
                    NumReadIC += Next

                # Global Accumulator Activation
                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = (Parameter.LayerIMG[i] ** 2) * Parameter.LayerInChan[i+1] / Parameter.GlobalAccParr

                # Global Buffer Activation
                NumOpBuffer = (Parameter.LayerIMG[i] ** 2) * Parameter.InputPrecesion * (Parameter.Kernel ** 2)
                NumOpBuffer *= Parameter.LayerInChan[i] / (Parameter.SRAMCol * Area.NumG)

                if Parameter.IsPoolLayer[i] == 1:
                    NumReadPool = ((Parameter.LayerIMG[i] / 2) ** 2) * Parameter.LayerInChan[i+1]
                    NumReadPool /= Parameter.GlobalAccParr

                if Parameter.IsPoolLayer[i] == 0:
                    NumReadPool = 0

            if Parameter.LayerType[i] == 'FC':
                # Number of Subarrays
                NumT = math.ceil(Parameter.LayerInChan[i] / Parameter.SubRow) * 2 * Parameter.ColumnsPerWeight
                NumT *= math.ceil(Parameter.LayerInChan[i + 1] / Parameter.SubCol)
                # Number of PE Elements
                NumT = math.ceil(1.0 * NumT / Parameter.NumSubperPE)
                # Number of Tile Elements
                NumT = math.ceil(1.0 * NumT / Parameter.NumPEperTile)

                NumReadIC = Parameter.LayerInChan[i] * Parameter.InputPrecesion
                NumReadIC += Parameter.LayerInChan[i + 1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))
                NumReadIC /= (Parameter.GlobalBusWidth * NumT / Parameter.NumTilePerChip)

                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = Parameter.LayerInChan[i+1] / Parameter.GlobalAccParr

                NumOpBuffer = Parameter.InputPrecesion * Parameter.LayerInChan[i] / (Parameter.SRAMCol * Area.NumG)

            if Parameter.LayerType[i + 1] == 'END':
                NumOpBuffer += (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) \
                               * Parameter.LayerInChan[i+1] / Parameter.SRAMCol
            IntCtotL += NumReadIC * IntCL * (2 + Parameter.DelPrecision/Parameter.InputPrecesion)
            ChipAcctotL += ChipAccL * NumReadAcc * 2
            GlobalBuffertotL += GlobalBuffer * NumOpBuffer * 3
            PoolL += ChipAccL * NumReadPool

        Total += (IntCtotL + ChipAcctotL + GlobalBuffertotL + PoolL) * Parameter.DATANUM

        Total += TileL

    Chip_Latency = Total
    GlobalAccumulLatency += ChipAcctotL * Parameter.DATANUM
    GlobalInterConnection_Latency += IntCtotL * Parameter.DATANUM
    GlobalBufferLatency += GlobalBuffertotL * Parameter.DATANUM
    AccumulLatency += ChipAcctotL * Parameter.DATANUM
    InterConnection_Latency += IntCtotL * Parameter.DATANUM
    BufferLatency += GlobalBuffertotL * Parameter.DATANUM

def Tile():
    global Tile_Latency
    global TileBufferLatency
    global TileAccumulLatency
    global TileInterConnection_Latency
    global InterConnection_Latency
    global AccumulLatency
    global BufferLatency
    # Latency consumption by PEs
    PEL = PE()

    # InterConnect Latency
    IntCL =  InterConnect(Parameter.NumPEperTile, Parameter.BusToPET, Area.PE_height, Area.PE_width)

    # Accumulator Latency
    RCANUM = Parameter.ADCPrecision + math.ceil(Parameter.InputPrecesion / Parameter.InputEncoding)
    RCANUM += (Parameter.ColumnsPerWeight-1)
    TileAccL = Accumulate(Parameter.NumPEperTile, RCANUM, 0)

    # Tile Buffer Latency
    if Parameter.TileBuffer == 'SRAM':
        TileBuffL = SRAMBuffer(0)
        # Not used yet
    if Parameter.TileBuffer == 'DFF':
        TileBuffL = DFF()

    Total = 0
    IntCLatency = 0
    TileAccLatency = 0
    TileBuffLatency = 0
    if Parameter.Mode == 'Inference_Normal':
        for i in range(Parameter.NumLayer):
            if Parameter.LayerType[i] == 'Conv':
                if Parameter.LayerInChan[i] * Parameter.Kernel * Parameter.Kernel <= Parameter.SubRow:
                    # Number of Subarrays
                    NumT = math.ceil(Parameter.ColumnsPerWeight * Parameter.LayerInChan[i+1] * 2 / Parameter.SubCol)
                    # Number of PE Elements
                    NumT = math.ceil(1.0 * NumT/Parameter.NumSubperPE)

                if Parameter.LayerInChan[i] * Parameter.Kernel * Parameter.Kernel > Parameter.SubRow:
                    # Number of Subarrays
                    if Parameter.LayerType[i + 1] == 'Conv':
                        Parr = min(Parameter.SubRow / Parameter.LayerInChan[i],
                                   Parameter.SubCol / Parameter.LayerInChan[i + 1])
                    if Parameter.LayerType[i + 1] == 'FC':
                        Parr = min(Parameter.SubRow / Parameter.LayerInChan[i],
                                   Parameter.SubCol / Parameter.LayerInChan[i])
                    NumT = (Parameter.Kernel ** 2) * 2 * Parameter.ColumnsPerWeight
                    NumT = math.ceil(NumT/Parr)
                    # Number of PE Elements
                    NumT = math.ceil(1.0 * NumT / Parameter.NumSubperPE)

                # Tile HTree Activation
                if Parameter.BusToPET == 'HTree':
                    NumReadIC = (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * Parameter.InputPrecesion
                    NumReadIC *= Parameter.LayerInChan[i] / (Parameter.SubRow * NumT)
                    if Parameter.LayerType[i + 1] == 'Conv':
                        Next = (Parameter.LayerIMG[i + 1] ** 2) / (Parameter.SubRow * NumT)
                        Next *= (Parameter.Kernel ** 2) * Parameter.InputPrecesion * Parameter.LayerInChan[i + 1]
                        NumReadIC += Next

                    if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                        Next = Parameter.LayerInChan[i + 1] * Parameter.InputPrecesion / (Parameter.SubRow * NumT)
                        NumReadIC += Next

                # Tile Bus Activation
                if Parameter.BusToPET == 'Bus':
                    NumReadIC = (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * Parameter.InputPrecesion
                    NumReadIC *= Parameter.LayerInChan[i] / (Parameter.SubRow)
                    if Parameter.LayerType[i + 1] == 'Conv':
                        Next = (Parameter.LayerIMG[i + 1] ** 2) * (Parameter.Kernel ** 2) * Parameter.LayerInChan[i + 1]
                        Next *= (RCANUM +math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) / (Parameter.SubRow)
                        NumReadIC += Next

                    if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                        Next = Parameter.LayerInChan[i + 1] / Parameter.SubRow
                        Next *= (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))
                        NumReadIC += Next

                # Tile Accumulator Activation
                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = (Parameter.LayerIMG[i] ** 2) * Parameter.LayerInChan[i+1]
                    NumReadAcc /= Parameter.TileAccParr

                # Tile Buffer Activation
                NumOpBuffer = (Parameter.LayerIMG[i] ** 2) * Parameter.InputPrecesion * (Parameter.Kernel ** 2)
                NumOpBuffer *= Parameter.LayerInChan[i] / (Parameter.SRAMCol * Area.NumITBG)

            if Parameter.LayerType[i] == 'FC':
                # Number of Subarrays
                NumT = math.ceil(Parameter.LayerInChan[i] / Parameter.SubRow) * 2 * Parameter.ColumnsPerWeight
                NumT *= math.ceil(Parameter.LayerInChan[i + 1] / Parameter.SubCol)
                # Number of PE Elements
                NumT = math.ceil(1.0 * NumT / Parameter.NumSubperPE)

                if Parameter.BusToPET == 'HTree':
                    NumReadIC = Parameter.LayerInChan[i] * Parameter.InputPrecesion / (Parameter.SubRow * NumT)
                    Next = Parameter.LayerInChan[i + 1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))
                    Next /= (Parameter.SubRow * NumT)
                    NumReadIC += Next

                if Parameter.BusToPET == 'Bus':
                    NumReadIC = Parameter.LayerInChan[i] * Parameter.InputPrecesion / (Parameter.SubRow)
                    Next = Parameter.LayerInChan[i + 1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))
                    Next /= (Parameter.SubRow)
                    NumReadIC += Next

                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = Parameter.LayerInChan[i+1]
                    NumReadAcc /= Parameter.TileAccParr

                NumOpBuffer = Parameter.InputPrecesion * Parameter.LayerInChan[i] / (Parameter.SRAMCol * Area.NumITBG)

            if Parameter.LayerType[i + 1] == 'Conv':
                Next = (Parameter.LayerIMG[i + 1] ** 2) * (Parameter.Kernel ** 2)
                Next *= (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) * Parameter.LayerInChan[i + 1]
                Next /= (Parameter.SRAMCol * Area.NumOTBG)
                NumOpBuffer += Next
            if Parameter.LayerType[i + 1] == 'END' or Parameter.LayerType[i + 1] == 'FC':
                Next = (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) * Parameter.LayerInChan[i+1]
                Next /= (Parameter.SRAMCol * Area.NumOTBG)
                NumOpBuffer += Next
            IntCLatency += NumReadIC * IntCL #/ Parameter.SpeedUP[i]
            TileAccLatency += TileAccL * NumReadAcc #/ Parameter.SpeedUP[i]
            TileBuffLatency += TileBuffL * NumOpBuffer #/ Parameter.SpeedUP[i]
        Total += (IntCLatency + TileAccLatency + TileBuffLatency) * Parameter.DATANUM

        Total += PEL

    if Parameter.Mode == 'Onchip_Normal':
        for i in range(Parameter.NumLayer):
            NumReadIC = 1
            NumReadAcc = 1
            NumOpBuffer = 1

    if Parameter.Mode == 'Onchip_Parallel':
        for i in range(Parameter.NumLayer):
            if Parameter.LayerType[i] == 'Conv':
                if Parameter.LayerInChan[i] * Parameter.Kernel * Parameter.Kernel <= Parameter.SubRow:
                    # Number of Subarrays
                    NumT = math.ceil(Parameter.ColumnsPerWeight * Parameter.LayerInChan[i + 1] * 2 / Parameter.SubCol)
                    # Number of PE Elements
                    NumT = math.ceil(1.0 * NumT / Parameter.NumSubperPE)

                if Parameter.LayerInChan[i] * Parameter.Kernel * Parameter.Kernel > Parameter.SubRow:
                    # Number of Subarrays
                    if Parameter.LayerType[i + 1] == 'Conv':
                        Parr = min(Parameter.SubRow/Parameter.LayerInChan[i], Parameter.SubCol/Parameter.LayerInChan[i+1])
                    if Parameter.LayerType[i + 1] == 'FC':
                        Parr = min(Parameter.SubRow / Parameter.LayerInChan[i],Parameter.SubCol / Parameter.LayerInChan[i])
                    NumT = (Parameter.Kernel ** 2) * 2 * Parameter.ColumnsPerWeight
                    NumT = math.ceil(NumT / Parr)
                    # Number of PE Elements
                    NumT = math.ceil(1.0 * NumT / Parameter.NumSubperPE)

                # Tile HTree Activation
                if Parameter.BusToPET == 'HTree':
                    NumReadIC = (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * Parameter.InputPrecesion
                    NumReadIC *= Parameter.LayerInChan[i] / (Parameter.SubRow * NumT)
                    if Parameter.LayerType[i + 1] == 'Conv':
                        Next = (Parameter.LayerIMG[i + 1] ** 2) / (Parameter.SubRow * NumT)
                        Next *= (Parameter.Kernel ** 2) * Parameter.InputPrecesion * Parameter.LayerInChan[i + 1]
                        NumReadIC += Next

                    if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                        Next = Parameter.LayerInChan[i + 1] * Parameter.InputPrecesion / (Parameter.SubRow * NumT)
                        NumReadIC += Next

                # Tile Bus Activation
                if Parameter.BusToPET == 'Bus':
                    NumReadIC = (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * Parameter.InputPrecesion
                    NumReadIC *= Parameter.LayerInChan[i] / (Parameter.SubRow)
                    if Parameter.LayerType[i + 1] == 'Conv':
                        Next = (Parameter.LayerIMG[i + 1] ** 2) * (Parameter.Kernel ** 2) * Parameter.LayerInChan[i + 1]
                        Next *= (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) / (Parameter.SubRow)
                        NumReadIC += Next

                    if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                        Next = Parameter.LayerInChan[i + 1] / Parameter.SubRow
                        Next *= (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))
                        NumReadIC += Next

                # Tile Accumulator Activation
                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = (Parameter.LayerIMG[i] ** 2) * Parameter.LayerInChan[i + 1]
                    NumReadAcc /= Parameter.TileAccParr

                # Tile Buffer Activation
                NumOpBuffer = (Parameter.LayerIMG[i] ** 2) * Parameter.InputPrecesion * (Parameter.Kernel ** 2)
                NumOpBuffer *= Parameter.LayerInChan[i] / (Parameter.SRAMCol * Area.NumITBG)

            if Parameter.LayerType[i] == 'FC':
                # Number of Subarrays
                NumT = math.ceil(Parameter.LayerInChan[i] / Parameter.SubRow) * 2 * Parameter.ColumnsPerWeight
                NumT *= math.ceil(Parameter.LayerInChan[i + 1] / Parameter.SubCol)
                # Number of PE Elements
                NumT = math.ceil(1.0 * NumT / Parameter.NumSubperPE)

                if Parameter.BusToPET == 'HTree':
                    NumReadIC = Parameter.LayerInChan[i] * Parameter.InputPrecesion / (Parameter.SubRow * NumT)
                    Next = Parameter.LayerInChan[i + 1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))
                    Next /= (Parameter.SubRow * NumT)
                    NumReadIC += Next

                if Parameter.BusToPET == 'Bus':
                    NumReadIC = Parameter.LayerInChan[i] * Parameter.InputPrecesion / (Parameter.SubRow)
                    Next = Parameter.LayerInChan[i + 1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))
                    Next /= (Parameter.SubRow)
                    NumReadIC += Next

                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = Parameter.LayerInChan[i + 1]
                    NumReadAcc /= Parameter.TileAccParr

                NumOpBuffer = Parameter.InputPrecesion * Parameter.LayerInChan[i] / (Parameter.SRAMCol * Area.NumITBG)

            if Parameter.LayerType[i + 1] == 'Conv':
                Next = (Parameter.LayerIMG[i + 1] ** 2) * (Parameter.Kernel ** 2)
                Next *= (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) * Parameter.LayerInChan[i + 1]
                Next /= (Parameter.SRAMCol * Area.NumOTBG)
                NumOpBuffer += Next
            if Parameter.LayerType[i + 1] == 'END' or Parameter.LayerType[i + 1] == 'FC':
                Next = (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) * Parameter.LayerInChan[i + 1]
                Next /= (Parameter.SRAMCol * Area.NumOTBG)
                NumOpBuffer += Next
            IntCLatency += NumReadIC * IntCL * (2 + Parameter.DelPrecision/Parameter.InputPrecesion)
            TileAccLatency += TileAccL * NumReadAcc * 2
            TileBuffLatency += TileBuffL * NumOpBuffer * 3
        Total += (IntCLatency + TileAccLatency + TileBuffLatency) * Parameter.DATANUM
        Total += PEL

    Tile_Latency = Total
    TileInterConnection_Latency += IntCLatency * Parameter.DATANUM
    TileAccumulLatency += TileAccLatency * Parameter.DATANUM
    TileBufferLatency += TileBuffLatency * Parameter.DATANUM
    BufferLatency += TileBuffLatency * Parameter.DATANUM
    AccumulLatency += TileAccLatency * Parameter.DATANUM
    InterConnection_Latency += IntCLatency * Parameter.DATANUM

    return Total

def PE():
    global PE_Latency
    global PEBufferLatency
    global PEAccumulLatency
    global PEInterConnection_Latency
    global InterConnection_Latency
    global AccumulLatency
    global BufferLatency

    # Latency consumption by PEs
    SubL = SubArray()
    # InterConnect Latency
    IntCL =  InterConnect(Parameter.NumSubperPE, Parameter.BusToSubT, Area.Subarray_height, Area.Subarray_width)

    # Accumulator Latency
    RCANUM = Parameter.ADCPrecision + math.ceil(Parameter.InputPrecesion / Parameter.InputEncoding)
    RCANUM += (Parameter.ColumnsPerWeight-1)
    PEAccL = Accumulate(Parameter.NumSubperPE, RCANUM ,0)

    # Tile Buffer Latency
    if Parameter.TileBuffer == 'SRAM':
        PEBuffL = SRAMBuffer(0)
        # Not used yet
    if Parameter.TileBuffer == 'DFF':
        PEBuffL = DFF()

    Total = 0
    IntCLatency = 0
    PEAccLatency = 0
    PEBuffLatency = 0
    if Parameter.Mode == 'Inference_Normal':
        for i in range(Parameter.NumLayer):
            if Parameter.LayerType[i] == 'Conv':
                if Parameter.LayerInChan[i] * Parameter.Kernel * Parameter.Kernel <= Parameter.SubRow:
                    # Number of Subarrays
                    NumT = math.ceil(Parameter.ColumnsPerWeight * Parameter.LayerInChan[i+1] * 2 / Parameter.SubCol)

                if Parameter.LayerInChan[i] * Parameter.Kernel * Parameter.Kernel > Parameter.SubRow:
                    # Number of Subarrays
                    if Parameter.LayerType[i + 1] == 'Conv':
                        Parr = min(Parameter.SubRow / Parameter.LayerInChan[i],
                                   Parameter.SubCol / Parameter.LayerInChan[i + 1])
                    if Parameter.LayerType[i + 1] == 'FC':
                        Parr = min(Parameter.SubRow / Parameter.LayerInChan[i],
                                   Parameter.SubCol / Parameter.LayerInChan[i])
                    NumT = (Parameter.Kernel ** 2) * 2 * Parameter.ColumnsPerWeight
                    NumT = math.ceil(NumT/Parr)

                # PE HTree Activation
                if Parameter.BusToSubT == 'HTree':
                    NumReadIC = (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * Parameter.InputPrecesion
                    NumReadIC *= Parameter.LayerInChan[i] / (Parameter.SubRow * NumT)
                    if Parameter.LayerType[i + 1] == 'Conv':
                        Next = (Parameter.LayerIMG[i + 1] ** 2) * (Parameter.Kernel ** 2) * Parameter.InputPrecesion
                        Next *= Parameter.LayerInChan[i + 1] / (Parameter.SubRow * NumT)
                        NumReadIC += Next

                    if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                        Next = Parameter.LayerInChan[i + 1] * Parameter.InputPrecesion / (Parameter.SubRow * NumT)
                        NumReadIC += Next

                # PE Bus Activation
                if Parameter.BusToSubT == 'Bus':
                    NumReadIC = (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * Parameter.InputPrecesion
                    NumReadIC *= Parameter.LayerInChan[i] / Parameter.SubRow
                    if Parameter.LayerType[i + 1] == 'Conv':
                        Next = (Parameter.LayerIMG[i + 1] ** 2) * (Parameter.Kernel ** 2)
                        Next *= Parameter.InputPrecesion * Parameter.LayerInChan[i + 1] / Parameter.SubRow
                        NumReadIC += Next

                    if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                        NumReadIC += Parameter.LayerInChan[i + 1] * Parameter.InputPrecesion / Parameter.SubRow

                # PE Accumulator Activation
                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = (Parameter.LayerIMG[i] ** 2) * Parameter.LayerInChan[i+1]
                    NumReadAcc /= Parameter.PEAccParr

                # PE Buffer Activation
                NumOpBuffer = (Parameter.LayerIMG[i] ** 2) * Parameter.InputPrecesion * (Parameter.Kernel ** 2)
                NumOpBuffer *= Parameter.LayerInChan[i] / (Parameter.SRAMCol* Area.NumIPEBG)

            if Parameter.LayerType[i] == 'FC':
                # Number of Subarrays
                NumT = math.ceil(Parameter.LayerInChan[i] / Parameter.SubRow) * 2 * Parameter.ColumnsPerWeight
                NumT *= math.ceil(Parameter.LayerInChan[i + 1] / Parameter.SubCol)

                if Parameter.BusToSubT == 'HTree':
                    NumReadIC = Parameter.LayerInChan[i] * Parameter.InputPrecesion / (Parameter.SubRow * NumT)
                    Next = Parameter.LayerInChan[i + 1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))
                    Next /= (Parameter.SubRow * NumT)
                    NumReadIC += Next

                if Parameter.BusToSubT == 'Bus':
                    NumReadIC = Parameter.LayerInChan[i] * Parameter.InputPrecesion / (Parameter.SubRow)
                    Next = Parameter.LayerInChan[i + 1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))
                    Next /= (Parameter.SubRow)
                    NumReadIC += Next

                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = Parameter.SubCol / Parameter.ColumnsPerWeight / Parameter.PEAccParr

                NumOpBuffer = Parameter.InputPrecesion * Parameter.LayerInChan[i] / (Parameter.SRAMCol * Area.NumIPEBG)

            if Parameter.LayerType[i + 1] == 'Conv':
                Next = (Parameter.LayerIMG[i + 1] ** 2) * (Parameter.Kernel ** 2) / (Parameter.SRAMCol * Area.NumOPEBG)
                Next *= (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) * Parameter.LayerInChan[i + 1]
                NumOpBuffer += Next
            if Parameter.LayerType[i + 1] == 'END' or Parameter.LayerType[i + 1] == 'FC':
                Next = (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) * Parameter.LayerInChan[i+1]
                Next /= (Parameter.SRAMCol * Area.NumOPEBG)
                NumOpBuffer += Next

            IntCLatency += NumReadIC * IntCL
            PEAccLatency += PEAccL * NumReadAcc
            PEBuffLatency += PEBuffL * NumOpBuffer

        Total += (IntCLatency + PEAccLatency + PEBuffLatency) * Parameter.DATANUM

        Total += SubL

    if Parameter.Mode == 'Onchip_Normal':
        for i in range(Parameter.NumLayer):
            NumReadIC = 1
            NumReadAcc = 1
            NumOpBuffer = 1

    if Parameter.Mode == 'Onchip_Parallel':
        for i in range(Parameter.NumLayer):
            if Parameter.LayerType[i] == 'Conv':
                if Parameter.LayerInChan[i] * Parameter.Kernel * Parameter.Kernel <= Parameter.SubRow:
                    # Number of Subarrays
                    NumT = math.ceil(Parameter.ColumnsPerWeight * Parameter.LayerInChan[i + 1] * 2 / Parameter.SubCol)

                if Parameter.LayerInChan[i] * Parameter.Kernel * Parameter.Kernel > Parameter.SubRow:
                    # Number of Subarrays
                    if Parameter.LayerType[i + 1] == 'Conv':
                        Parr = min(Parameter.SubRow / Parameter.LayerInChan[i],
                                   Parameter.SubCol / Parameter.LayerInChan[i + 1])
                    if Parameter.LayerType[i + 1] == 'FC':
                        Parr = min(Parameter.SubRow / Parameter.LayerInChan[i],
                                   Parameter.SubCol / Parameter.LayerInChan[i])
                    NumT = (Parameter.Kernel ** 2) * 2 * Parameter.ColumnsPerWeight
                    NumT = math.ceil(NumT / Parr)

                # PE HTree Activation
                if Parameter.BusToSubT == 'HTree':
                    NumReadIC = (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * Parameter.InputPrecesion
                    NumReadIC *= Parameter.LayerInChan[i] / (Parameter.SubRow * NumT)
                    if Parameter.LayerType[i + 1] == 'Conv':
                        Next = (Parameter.LayerIMG[i + 1] ** 2) * (Parameter.Kernel ** 2) * Parameter.InputPrecesion
                        Next *= Parameter.LayerInChan[i + 1] / (Parameter.SubRow * NumT)
                        NumReadIC += Next

                    if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                        Next = Parameter.LayerInChan[i + 1] * Parameter.InputPrecesion / (Parameter.SubRow * NumT)
                        NumReadIC += Next

                # PE Bus Activation
                if Parameter.BusToSubT == 'Bus':
                    NumReadIC = (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * Parameter.InputPrecesion
                    NumReadIC *= Parameter.LayerInChan[i] / Parameter.SubRow
                    if Parameter.LayerType[i + 1] == 'Conv':
                        Next = (Parameter.LayerIMG[i + 1] ** 2) * (Parameter.Kernel ** 2)
                        Next *= Parameter.InputPrecesion * Parameter.LayerInChan[i + 1] / Parameter.SubRow
                        NumReadIC += Next

                    if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                        NumReadIC += Parameter.LayerInChan[i + 1] * Parameter.InputPrecesion / Parameter.SubRow

                # PE Accumulator Activation
                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = (Parameter.LayerIMG[i] ** 2) * Parameter.LayerInChan[i + 1]
                    NumReadAcc /= Parameter.PEAccParr

                # PE Buffer Activation
                NumOpBuffer = (Parameter.LayerIMG[i] ** 2) * Parameter.InputPrecesion * (Parameter.Kernel ** 2)
                NumOpBuffer *= Parameter.LayerInChan[i] / (Parameter.SRAMCol * Area.NumIPEBG)

            if Parameter.LayerType[i] == 'FC':
                # Number of Subarrays
                NumT = math.ceil(Parameter.LayerInChan[i] / Parameter.SubRow) * 2 * Parameter.ColumnsPerWeight
                NumT *= math.ceil(Parameter.LayerInChan[i + 1] / Parameter.SubCol)

                if Parameter.BusToSubT == 'HTree':
                    NumReadIC = Parameter.LayerInChan[i] * Parameter.InputPrecesion / (Parameter.SubRow * NumT)
                    Next = Parameter.LayerInChan[i + 1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))
                    Next /= (Parameter.SubRow * NumT)
                    NumReadIC += Next

                if Parameter.BusToSubT == 'Bus':
                    NumReadIC = Parameter.LayerInChan[i] * Parameter.InputPrecesion / (Parameter.SubRow)
                    Next = Parameter.LayerInChan[i + 1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))
                    Next /= (Parameter.SubRow)
                    NumReadIC += Next

                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = Parameter.SubCol / Parameter.ColumnsPerWeight / Parameter.PEAccParr

                NumOpBuffer = Parameter.InputPrecesion * Parameter.LayerInChan[i] / (Parameter.SRAMCol * Area.NumIPEBG)

            if Parameter.LayerType[i + 1] == 'Conv':
                Next = (Parameter.LayerIMG[i + 1] ** 2) * (Parameter.Kernel ** 2) / (Parameter.SRAMCol * Area.NumOPEBG)
                Next *= (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) * Parameter.LayerInChan[i + 1]
                NumOpBuffer += Next
            if Parameter.LayerType[i + 1] == 'END' or Parameter.LayerType[i + 1] == 'FC':
                Next = (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) * Parameter.LayerInChan[i + 1]
                Next /= (Parameter.SRAMCol * Area.NumOPEBG)
                NumOpBuffer += Next

            IntCLatency += NumReadIC * IntCL * (2 + Parameter.DelPrecision/Parameter.InputPrecesion)
            PEAccLatency += PEAccL * NumReadAcc * 2
            PEBuffLatency += PEBuffL * NumOpBuffer * 3

        Total += (IntCLatency + PEAccLatency + PEBuffLatency) * Parameter.DATANUM

        Total += SubL
    PEBufferLatency += PEBuffLatency * Parameter.DATANUM
    PEAccumulLatency += PEAccLatency * Parameter.DATANUM
    PEInterConnection_Latency += IntCLatency * Parameter.DATANUM
    BufferLatency += PEBuffLatency * Parameter.DATANUM
    AccumulLatency += PEAccLatency * Parameter.DATANUM
    InterConnection_Latency += IntCLatency * Parameter.DATANUM

    return Total


def SubArray():
    global Subarray_Latency
    global ADCLatency
    global MemoryArrayLatency
    global DriverLatency
    global LevelShifterLatency
    global ShiftAdderLatency
    global MUXLatency
    global UpdateLatency

    if Parameter.Mode == 'Inference_Normal':
        ML = MemoryArray(0)  # This is included in Other circuits.
        DL = Driver(Parameter.SubCol * 0.06e-15)
        LL = Levelshifter(Parameter.SubCol * 0.06e-15)
        if Parameter.ColumnsPerADC > 1:
            MUXL = MUX(Parameter.SubRow * 0.08e-15)
        ADCL = ADC()
        RCABit = math.ceil(Parameter.InputPrecesion/Parameter.InputEncoding) + Parameter.ADCPrecision
        RCABit += (Parameter.ColumnsPerWeight-1)
        ShAdL = ShiftAdd(RCABit, 0)
        Subarray_L = 0
        TotADCL = 0
        TotLL = 0
        TotDL = 0
        TotShAdL = 0
        TotML = 0
        TotMUXL = 0

        for i in range(Parameter.NumLayer):
            if Parameter.LayerType[i] == 'Conv':
                # Driver Operation Number
                InpCy = Parameter.InputPrecesion / Parameter.InputEncoding

                NumOpD = InpCy * (Parameter.LayerIMG[i] ** 2)

                # Level Shifter Operation Number
                NumOpL = 0  # This is Inferencing Moder

                # ADC Operation Numbers
                NumOpADC = InpCy * Parameter.ColumnsPerADC * (Parameter.LayerIMG[i] ** 2)
                # In NeuroSim InpCy is not considered in ADC Operation numbers. To fit with NeuroSim remove InpCy

                # Shift Adder Operation Numbers
                if Parameter.InputPrecesion != Parameter.InputEncoding or Parameter.ColumnsPerWeight != 0:
                    NumOpShAd = InpCy * Parameter.ColumnsPerWeight
                    NumOpShAd *= (Parameter.LayerIMG[i] ** 2)

            if Parameter.LayerType[i] == 'FC':
                # Driver Operation Number
                InpCy = Parameter.InputPrecesion / Parameter.InputEncoding

                NumOpD = InpCy

                # Level Shifter Operation Number
                NumOpL = 0  # This is Inferencing Moder

                # ADC Operation Numbers
                NumOpADC = InpCy * Parameter.ColumnsPerADC

                # Shift Adder Operation Numbers
                if Parameter.InputPrecesion != Parameter.InputEncoding or Parameter.ColumnsPerWeight != 0:
                    NumOpShAd = InpCy * Parameter.ColumnsPerWeight

            Subarray_L += ((DL * NumOpD) + (LL * NumOpL) + (ADCL * NumOpADC) + (ShAdL * NumOpShAd)) * Parameter.DATANUM #/ Parameter.SpeedUP[i]
            # Memory Read Time
            Subarray_L += ML * NumOpD * Parameter.DATANUM #/ Parameter.SpeedUP[i]
            # Speed Up is For NeuroSim Comparison

            TotLL += (LL * NumOpL) * Parameter.DATANUM
            TotADCL += (ADCL * NumOpADC) * Parameter.DATANUM
            TotDL += (DL * NumOpD) * Parameter.DATANUM
            TotShAdL += (ShAdL * NumOpShAd) * Parameter.DATANUM
            TotML += (ML * NumOpD) * Parameter.DATANUM
            TotMUXL += MUXL * NumOpD * Parameter.DATANUM
    if Parameter.Mode == 'Onchip_Normal':
        # FF Side
        ME = MemoryArray()
        DE = Driver()
        LE = Levelshifter()
        if Parameter.ColumnsPerADC > 1:
            MUXE = MUX()
        ADCE = ADC()
        ShAdE = ShiftAdd()

        Subarray_E = 1

    if Parameter.Mode == 'Onchip_Parallel':
        ML = MemoryArray(0)  # This is included in Other circuits.
        DL = Driver(Parameter.SubCol * 0.06e-15)
        LL = Levelshifter(Parameter.SubCol * 0.06e-15)
        if Parameter.ColumnsPerADC > 1:
            MUXL = MUX(Parameter.SubRow * 0.08e-15)
        ADCL = ADC()
        RCABit = math.ceil(Parameter.InputPrecesion / Parameter.InputEncoding) + Parameter.ADCPrecision
        RCABit += (Parameter.ColumnsPerWeight - 1)
        ShAdL = ShiftAdd(RCABit, 0)
        Subarray_L = 0
        TotADCL = 0
        TotLL = 0
        TotDL = 0
        TotShAdL = 0
        TotML = 0
        TotMUXL = 0

        for i in range(Parameter.NumLayer):
            if Parameter.LayerType[i] == 'Conv':
                # Driver Operation Number
                InpCy = Parameter.InputPrecesion / Parameter.InputEncoding
                InpCy2 = Parameter.DelPrecision / Parameter.DelEn

                NumOpD = InpCy * (Parameter.LayerIMG[i] ** 2)
                if Parameter.LayerType[i + 1] == 'Conv':
                    NumOpD += InpCy2 * (Parameter.LayerIMG[i+1] ** 2)
                if Parameter.LayerType[i + 1] == 'FC':
                    NumOpD += InpCy2 * (Parameter.LayerIMG[i+1])

                # Level Shifter Operation Number
                NumOpL = 0  # We will consider it with Weight Update Latency

                # ADC Operation Numbers
                NumOpADC = InpCy * Parameter.ColumnsPerADC * (Parameter.LayerIMG[i] ** 2)
                if Parameter.LayerType[i + 1] == 'Conv':
                    NumOpADC += InpCy2 * Parameter.RowsPerADC * (Parameter.LayerIMG[i+1] ** 2)
                if Parameter.LayerType[i + 1] == 'FC':
                    NumOpADC += InpCy2 * (Parameter.LayerIMG[i+1])

                # Shift Adder Operation Numbers
                if Parameter.InputPrecesion != Parameter.InputEncoding or Parameter.ColumnsPerWeight != 0:
                    NumOpShAd = InpCy * Parameter.ColumnsPerWeight
                    NumOpShAd *= (Parameter.LayerIMG[i] ** 2)

                if Parameter.DelPrecision != Parameter.DelEn:   # Columns per weight should be 1 in Onchip Parallel
                    if Parameter.LayerType[i + 1] == 'Conv':
                        NumOpShAd += InpCy2 * (Parameter.LayerIMG[i+1] ** 2)
                    if Parameter.LayerType[i + 1] == 'FC':
                        NumOpShAd += InpCy2 * (Parameter.LayerIMG[i + 1])

            if Parameter.LayerType[i] == 'FC':
                # Driver Operation Number
                InpCy = Parameter.InputPrecesion / Parameter.InputEncoding
                InpCy2 = Parameter.DelPrecision / Parameter.DelEn

                NumOpD = InpCy + InpCy2

                # Level Shifter Operation Number
                NumOpL = 0  #

                # ADC Operation Numbers
                NumOpADC = InpCy * Parameter.ColumnsPerADC
                NumOpADC += InpCy2 * Parameter.RowsPerADC

                # Shift Adder Operation Numbers
                if Parameter.InputPrecesion != Parameter.InputEncoding or Parameter.ColumnsPerWeight != 0:
                    NumOpShAd = InpCy * Parameter.ColumnsPerWeight

                if Parameter.DelPrecision != Parameter.DelEn:
                    NumOpShAd += InpCy2


            Subarray_L += ((DL * NumOpD) + (LL * NumOpL) + (ADCL * NumOpADC) + (ShAdL * NumOpShAd)) * Parameter.DATANUM
            # Memory Read Time
            Subarray_L += ML * NumOpD * Parameter.DATANUM + MUXL * NumOpD * Parameter.DATANUM

            TotLL += (LL * Parameter.IMGSIZE * Parameter.IMGSIZE) * Parameter.DATANUM
            TotADCL += (ADCL * NumOpADC) * Parameter.DATANUM
            TotDL += (DL * NumOpD) * Parameter.DATANUM
            TotShAdL += (ShAdL * NumOpShAd) * Parameter.DATANUM
            TotML += (ML * NumOpD) * Parameter.DATANUM
            TotMUXL += MUXL * NumOpD * Parameter.DATANUM
        if Parameter.Onchip_ParMode == 'LTP_only':
            UpdateLatency += Parameter.FullTimeLTP * Parameter.UpdateMax * (Parameter.IMGSIZE ** 2) * Parameter.DATANUM
        if Parameter.Onchip_ParMode == 'LTD_only':
            UpdateLatency += Parameter.FUllTimeLTD * Parameter.UpdateMax * (Parameter.IMGSIZE ** 2) * Parameter.DATANUM
        if Parameter.Onchip_ParMode == 'LTPLTD_both':
            UpdateLatency += Parameter.FUllTimeLTD * Parameter.UpdateMax * (Parameter.IMGSIZE ** 2) * Parameter.DATANUM * 0.5
            UpdateLatency += Parameter.FullTimeLTP * Parameter.UpdateMax * (Parameter.IMGSIZE ** 2) * Parameter.DATANUM * 0.5


        Subarray_L += UpdateLatency

    ADCLatency += TotADCL
    DriverLatency += TotDL
    LevelShifterLatency += TotLL
    MemoryArrayLatency += TotML
    ShiftAdderLatency += TotShAdL
    MUXLatency += TotMUXL

    Subarray_Latency = Subarray_L
    return  Subarray_L

def Driver(CapOut):
    # Single Driver Latency
    if Parameter.HighVoltage > Parameter.VDD:  # Use an AND gate besides Levelshifter for Driver
        # DFF Latency
        CapIn = Parameter.GateCapN + Parameter.GateCapP
        LatencyD = DFF(CapIn)

        # Using Horowitz
        R1 = Parameter.VDD * Parameter.EFFR / (Parameter.PONCurrS)
        Cap1 = Parameter.JuncCapN + Parameter.JuncCapP * 2 + 2 * Parameter.GateCapP + Parameter.GateCapN
        Latency1 = horowitz(R1, Cap1, 1/LatencyD)

        # Using Elmore Delay (Connected to the Array model)
        R2 =  Parameter.VDD * Parameter.EFFR / (Parameter.NONCurrS)
        Cap2 = Parameter.JuncCapN + Parameter.JuncCapP * 2 + Parameter.SubCol * 0.06e-15 + CapOut
        Latency2 = horowitz(R2, Cap2,1/Latency1)

        Latency = LatencyD + Latency1 + Latency2

    if Parameter.HighVoltage <= Parameter.VDD:       # In this case, we can adopt WLDriver Suggested by S.Yu
        # DFF Latency
        CapIn = Parameter.GateCapN * (1 + 2) + Parameter.GateCapP * 2
        LatencyD = DFF(CapIn)

        # Driver Latency
        RArray = Parameter.SubCol * Parameter.RA
        RTG = 0.5 * Parameter.VDD * Parameter.EFFR / (Parameter.NONCurrS)
        CArray = Parameter.SubCol * 0.06e-15
        Latency1 = (2*RTG+RArray)*(CArray/2) * 0.7 * 0.9
        Latency1 += (2 * RTG) * (CArray/2+2*Parameter.JuncCapP + Parameter.JuncCapN) * 0.7 * 0.9
        Latency1 += RTG * (6 * Parameter.JuncCapP + 3 * Parameter.JuncCapN) * 0.7 * 0.9

        Latency  = Latency1 + LatencyD

    return Latency

def PWM():
    # Will Be added Later
    return 0

def Levelshifter(CapOut):
    # Single Levelshifter Latency
    L = math.sqrt(Parameter.HighVoltage/3.3) * (270 / Parameter.Tech)      # Channel Length of Levelshifter

    R1 = Parameter.VDD * Parameter.EFFR / (Parameter.NONCurrS * 15)
    Cap1 = (Parameter.JuncCapP * 20 + Parameter.JuncCapN * 15) + Parameter.GateCapN * 32 * L
    Latency1 = horowitz(R1, Cap1, 1e12)

    R2 =  Parameter.VDD * Parameter.EFFR * 8 / (Parameter.NONCurrS * 32)
    Latency2 = R2 * (Parameter.JuncCapP *10+Parameter.JuncCapN*32 + Parameter.GateCapN * 146 * L + CapOut) * 0.7 * 0.9

    R3 = Parameter.VDD * Parameter.EFFR * 16 / (Parameter.NONCurrS * 82)
    Latency3 = R3 * (6e-17 * Parameter.SubCol / 2 + Parameter.JuncCapP * 64 + Parameter.JuncCapN * 82) * 0.7 * 0.9
    Latency3 += (R3 + Parameter.SubCol * Parameter.RA) * (6e-17 * Parameter.SubCol / 2) * 0.7 * 0.9

    Latency = Latency1 + Latency2 + Latency3
    return Latency

def MemoryArray(CapOut):
    #Latency By Array is considered in the Peripheral Circuits
    return Parameter.ReadTimePerCycle

def MUX(CapOut):
    # Single MUX Latency
    if Parameter.ColumnsPerADC == 1:
        return 0

    NumInNOR = math.ceil(math.log2(Parameter.ColumnsPerADC) / 2)

    #(We Support only for Parameter.ColumnsPerADC <= 8)
    # MUX Decoder Latency
    R1 = Parameter.VDD * Parameter.EFFR / (Parameter.NONCurrS)
    Cap1 = (Parameter.JuncCapN + Parameter.JuncCapP * 2 + Parameter.GateCapN  + Parameter.GateCapP)
    Latency1 =  horowitz(R1, Cap1, 1e12)

    R2 = Parameter.VDD * Parameter.EFFR / (Parameter.PONCurrS)
    Cap2 = (Parameter.JuncCapN + Parameter.JuncCapP * 2 + Parameter.GateCapN * 1 + Parameter.GateCapP * 2)
    Latency2 = horowitz(R2, Cap2, 1/Latency1)

    R3 = Parameter.VDD * Parameter.EFFR / (Parameter.NONCurrS)
    Cap3 = (Parameter.JuncCapN + Parameter.JuncCapP * 2 + Parameter.GateCapN + Parameter.GateCapP * NumInNOR * 2)
    Latency3 = horowitz(R3, Cap3, 1/Latency2)

    R4 = Parameter.VDD * Parameter.EFFR / (Parameter.PONCurrS * 2)
    Cap4 = Parameter.ColumnsPerADC * (0.12e-15) / 2 + Parameter.JuncCapN * 1 + Parameter.JuncCapP * 2 * NumInNOR
    Latency4 = R4 * Cap4 * 0.7 * 0.9

    R5 = R4 + Parameter.ColumnsPerADC * Parameter.RA
    Cap5 = Parameter.ColumnsPerADC * (0.12e-15) / 2
    Latency5 = R5 * Cap5 * 0.7 * 0.9

    LatencyDEC = Latency1 + Latency2 + Latency3 + Latency4 + Latency5

    if Parameter.ColumnsPerADC == 2:
        R1 = 0.5 * Parameter.VDD * Parameter.EFFR / (Parameter.NONCurrS)
        Cap1 = (Parameter.JuncCapP * 4 + Parameter.JuncCapN * 2 + Parameter.SubRow * 0.08e-15 / 2)
        Latency1 = R1 * Cap1 * 0.7 * 0.9
        R2 = R1 + Parameter.SubRow * Parameter.RA
        Cap2 = Parameter.SubRow * 0.08e-15 / 2
        Latency2 = R2 * Cap2 * 0.7 * 0.9
        Latency3 = 0
        Latency4 = 0

    if Parameter.ColumnsPerADC == 4:
        R1 = 0.5 * Parameter.VDD * Parameter.EFFR / (Parameter.NONCurrS)
        Cap1 = (Parameter.JuncCapP * 6 + Parameter.JuncCapN * 3)
        Latency1 = R1 * Cap1 * 0.7 * 0.9
        R2 = 0.5 * Parameter.VDD * Parameter.EFFR / (Parameter.NONCurrS) + R1
        Cap2 = (Parameter.JuncCapP * 4 + Parameter.JuncCapN * 2 + Parameter.SubRow * 0.08e-15 / 2)
        Latency2 = R2 * Cap2 * 0.7 * 0.9
        R3 = R2 + Parameter.SubRow * Parameter.RA
        Cap3 = Parameter.SubRow * 0.08e-15 / 2
        Latency3 = R3 * Cap3 * 0.7 * 0.9
        Latency4 = 0


    if Parameter.ColumnsPerADC == 8:
        R1 = 0.5 * Parameter.VDD * Parameter.EFFR / (Parameter.NONCurrS)
        Cap1 = (Parameter.JuncCapP * 6 + Parameter.JuncCapN * 3)
        Latency1 = R1 * Cap1 * 0.7 * 0.9
        R2 = 0.5 * Parameter.VDD * Parameter.EFFR / (Parameter.NONCurrS) + R1
        Cap2 = (Parameter.JuncCapP * 6 + Parameter.JuncCapN * 3)
        Latency2 = R2 * Cap2 * 0.7 * 0.9
        R3 = R2 + 0.5 * Parameter.VDD * Parameter.EFFR / (Parameter.NONCurrS)
        Cap3 = (Parameter.JuncCapP * 4 + Parameter.JuncCapN * 2 + Parameter.SubRow * 0.08e-15 / 2)
        Latency3 = R3 * Cap3 * 0.7 * 0.9
        R4 = R3 + Parameter.SubRow * Parameter.RA
        Cap4 = Parameter.SubRow * 0.08e-15 / 2
        Latency4 = R4 * Cap4 * 0.7 * 0.9

    Latency = Latency1 + Latency2 + Latency3 + Latency4 + LatencyDEC
    return Latency


def ADC():
    # Single ADC Latency
    #Calculating Unit ADC Area
    if Parameter.ADCType == 'SAR':  # Here we adopted Formula From NeuroSim(S. Yu) directly
        Latency = Parameter.ADCPrecision * Parameter.ADCCycle

    if Parameter.ADCType == 'Flash':
        Latency = 0     # Here will be added later

    return Latency

def ShiftAdd(RCABit, CapOut):
    # Single DFF Latency
    Latency1 = DFF(Parameter.GateCapN * 2 + Parameter.GateCapP * 2)

    # Single Full Adder Latency
    Latency2 = Accumulate(2, RCABit, 0)
    # Single Shift Adder Latency
    Latency = Latency1 + Latency2

    return Latency

def Accumulate(NumtoAcc, RCANUM, CapOut):
    # Single Full Adder Latency

    R1 = Parameter.VDD * Parameter.EFFR / (Parameter.PONCurrS * 1)
    Cap1 = (Parameter.JuncCapN + Parameter.JuncCapP * 2 + Parameter.GateCapN * 3 + Parameter.GateCapP * 3)
    Latency1 = horowitz(R1, Cap1, 1e12)

    R2 = Parameter.VDD * Parameter.EFFR / (Parameter.NONCurrS * 0.5)
    Cap2 = (Parameter.JuncCapN + Parameter.JuncCapP * 2 + Parameter.GateCapN * 1 + Parameter.GateCapP * 1)
    Latency2 = horowitz(R2, Cap2, 1/Latency1)

    R3 = Parameter.VDD * Parameter.EFFR / (Parameter.PONCurrS * 1)
    Cap3 = (Parameter.JuncCapN + Parameter.JuncCapP * 2 + Parameter.GateCapN * 3 + Parameter.GateCapP * 3)
    Latency3 = horowitz(R3, Cap3, 1/Latency2)

    R4 = Parameter.VDD * Parameter.EFFR / (Parameter.NONCurrS * 0.5)
    Cap4 = (Parameter.JuncCapN + Parameter.JuncCapP * 2 + Parameter.GateCapN * 1 + Parameter.GateCapP * 1)
    Latency4 = horowitz(R4, Cap4, 1/Latency3)

    R5 = Parameter.VDD * Parameter.EFFR / (Parameter.PONCurrS * 1)
    Cap5 = (Parameter.JuncCapN + Parameter.JuncCapP * 2 + Parameter.GateCapN * 2 + Parameter.GateCapP * 2)
    Latency5 = horowitz(R5, Cap5, 1/Latency4)

    R6 = Parameter.VDD * Parameter.EFFR / (Parameter.NONCurrS * 0.5)
    Cap6 = (Parameter.JuncCapN + Parameter.JuncCapP * 2) + CapOut
    Latency6 = horowitz(R6, Cap6, 1/Latency5)

    LatencyPathSum = Latency1 + Latency2 + Latency3 + Latency4 + Latency5 + Latency6

    LatencyPathCarry = Latency1 + Latency2

    # Single Ripple Carray Adder Latency
    Latency =  LatencyPathSum + LatencyPathCarry * (RCANUM - 1)

    Latency *= math.ceil(math.log2(NumtoAcc))

    return Latency

def SRAMBuffer(CapOut):
    # Single Decoder Latency
    NumInNOR = math.ceil(math.log2(Parameter.SRAMRow) / 2)

    # Decoder Latency
    R1 = Parameter.VDD * Parameter.EFFR / (Parameter.NONCurrS)
    Cap1 = (Parameter.JuncCapN + Parameter.JuncCapP * 2 + Parameter.GateCapN  + Parameter.GateCapP)
    Latency1 =  horowitz(R1, Cap1, 1e12)

    R2 = Parameter.VDD * Parameter.EFFR / (Parameter.PONCurrS)
    Cap2 = (Parameter.JuncCapN + Parameter.JuncCapP * 2 + Parameter.GateCapN * 1 + Parameter.GateCapP * 2)
    Latency2 = horowitz(R2, Cap2, 1/Latency1)

    R3 = Parameter.VDD * Parameter.EFFR / (Parameter.NONCurrS)
    Cap3 = (Parameter.JuncCapN + Parameter.JuncCapP * 2 + Parameter.GateCapN + Parameter.GateCapP * NumInNOR * 2)
    Latency3 = horowitz(R3, Cap3, 1/Latency2)

    R4 = Parameter.VDD * Parameter.EFFR / (Parameter.PONCurrS * 2)
    Cap4 = Parameter.SRAMCol * 8e-17 / 2 + Parameter.JuncCapN * 1 + Parameter.JuncCapP * 2 * NumInNOR
    Latency4 = R4 * Cap4 * 0.7 * 0.9

    R5 = R4 + Parameter.SRAMCol * Parameter.RA
    Cap5 = Parameter.SRAMCol * 8e-17 / 2
    Latency5 = R5 * Cap5 * 0.7 * 0.9

    LatencyDEC = Latency1 + Latency2 + Latency3 + Latency4 + Latency5

    # Cell Precharge Latency
    RP = Parameter.VDD * Parameter.EFFR / (Parameter.PONCurrS * 6)
    CapP = Parameter.SRAMRow * Parameter.JuncCapN
    LatencyP = horowitz(RP, CapP, 1e12)

    # Single Cell Read Latency
    ReadLatency = LatencyDEC + LatencyP + 193e-12

    # Single Cell Write Latency
    WriteLatency = LatencyDEC + LatencyP + 350e-12

    Latency = ReadLatency + WriteLatency

    return Latency

def DFF(CapOut):
    # Single DFF Delay
    R1 = Parameter.VDD * Parameter.EFFR / (Parameter.PONCurrS * 2)
    Cap1 = (Parameter.JuncCapN + Parameter.JuncCapP * 2 + Parameter.GateCapN * 2 + Parameter.GateCapP * 2)
    Latency1 = horowitz(R1, Cap1, 1e12)

    R2 = Parameter.VDD * Parameter.EFFR / (Parameter.NONCurrS * 0.5)
    Cap2 = (Parameter.JuncCapN + Parameter.JuncCapP * 2 + Parameter.GateCapN * 2 + Parameter.GateCapP * 2)
    Latency2 = horowitz(R2, Cap2, 1/Latency1)

    R3 = Parameter.VDD * Parameter.EFFR / (Parameter.PONCurrS * 1)
    Cap3 = (Parameter.JuncCapN + Parameter.JuncCapP * 2 + Parameter.GateCapN * 2 + Parameter.GateCapP * 2) + CapOut
    Latency3 = horowitz(R3, Cap3, 1/Latency2)

    Latency = Latency1 + Latency2 + Latency3
    return Latency

def InterConnect(NumElements, Type, Height, Width):
    if Type == 'HTree':
        Latency = 0
        # Minimum Inverter CapSize
        if Parameter.Tech == 45:
            MININVCAP = 0.044e-15 + 0.085e-15                # Only Gate Cap (measured with SPICE)
            MININVCAPTO = MININVCAP + 0.049e-15 + 0.11e-15      # Capacitance of (Gate + Drain)

        #Inter Connect Repeater Size and Minimum Distance for Repeater
        ResonRep = (Parameter.CONSTEFFR*Parameter.VDD/Parameter.NONCurr)
        ResonRep += (Parameter.CONSTEFFR * Parameter.VDD / (Parameter.PNRATIO * Parameter.PONCurr))
        MINDIST = math.sqrt(2 * ResonRep *MININVCAPTO / (Parameter.UnitWiCap * Parameter.UnitWiRes))

        numStage = math.ceil(math.log2(NumElements))
        WireLenH = Height * math.pow(2, (numStage)/2)
        WireLenW = Width * math.pow(2, (numStage)/2)

        # STARTs From the Height
        numREP = 1e-3 * WireLenH / MINDIST
        if numREP >= 1:
            UnitLatency = 0.7 * (ResonRep * (MININVCAPTO + Parameter.UnitWiCap * MINDIST)) / MINDIST
            UnitLatency += 0.7 * (0.5 * Parameter.UnitWiRes * MINDIST * Parameter.UnitWiCap * MINDIST) / MINDIST
            UnitLatency += 0.7 * (Parameter.UnitWiRes * MINDIST * MININVCAP) / MINDIST
        if numREP < 1:
            UnitLatency = 0.7 * Parameter.UnitWiCap * Parameter.UnitWiRes * MINDIST
        Latency += 1e-3 * WireLenH * UnitLatency

        # Repeats with Stages
        for i in range(0, int((numStage) / 2)):
            WireLenW /= 2
            numREP = 1e-3 * WireLenW / MINDIST
            if numREP >= 1:
                UnitLatency = 0.7 * (ResonRep * (MININVCAPTO + Parameter.UnitWiCap*MINDIST))/MINDIST
                UnitLatency += 0.7 * (0.5 * Parameter.UnitWiRes * MINDIST * Parameter.UnitWiCap * MINDIST)/MINDIST
                UnitLatency += 0.7 * (Parameter.UnitWiRes * MINDIST * MININVCAP)/MINDIST
            if numREP < 1:
                UnitLatency = 0.7 * Parameter.UnitWiCap * Parameter.UnitWiRes * MINDIST
            Latency += 1e-3 * WireLenW * UnitLatency

            WireLenH /= 2
            numREP = 1e-3 * WireLenH / MINDIST
            if numREP >= 1:
                UnitLatency = 0.7 * (ResonRep * (MININVCAPTO + Parameter.UnitWiCap*MINDIST))/MINDIST
                UnitLatency += 0.7 * (0.5 * Parameter.UnitWiRes * MINDIST * Parameter.UnitWiCap * MINDIST)/MINDIST
                UnitLatency += 0.7 * (Parameter.UnitWiRes * MINDIST * MININVCAP)/MINDIST
            if numREP < 1:
                UnitLatency = 0.7 * Parameter.UnitWiCap * Parameter.UnitWiRes * MINDIST
            Latency += 1e-3 * WireLenH * UnitLatency

    if Type == 'Bus':
        Latency = 0
        # Minimum Inverter CapSize
        if Parameter.Tech == 45:
            MININVCAP = 0.044e-15 + 0.085e-15                # Only Gate Cap (measured with SPICE)
            MININVCAPTO = MININVCAP + 0.049e-15 + 0.11e-15      # Capacitance of (Gate + Drain)

        #Inter Connect Repeater Size and Minimum Distance for Repeater
        ResonRep = (Parameter.CONSTEFFR*Parameter.VDD/Parameter.NONCurr)
        ResonRep += (Parameter.CONSTEFFR * Parameter.VDD / (Parameter.PNRATIO * Parameter.PONCurr))
        MINDIST = math.sqrt(2 * ResonRep *MININVCAPTO / (Parameter.UnitWiCap * Parameter.UnitWiRes))

        WireLength = math.floor(math.sqrt(NumElements)) * min(Height, Width)
        numREP = 1e-3 * WireLength / MINDIST

        if numREP >= 1:
            UnitLatency = 0.7 * (ResonRep * (MININVCAPTO + Parameter.UnitWiCap * MINDIST)) / MINDIST
            UnitLatency += 0.7 * (0.5 * Parameter.UnitWiRes * MINDIST * Parameter.UnitWiCap * MINDIST) / MINDIST
            UnitLatency += 0.7 * (Parameter.UnitWiRes * MINDIST * MININVCAP) / MINDIST
        if numREP < 1:
            UnitLatency = 0.7 * Parameter.UnitWiCap * Parameter.UnitWiRes * MINDIST
        Latency += 1e-3 * WireLength * UnitLatency

    return Latency

def Print(): # Print Areas
    global Total
    # Module Latency
    global Chip_Latency 
    global Tile_Latency 
    global PE_Latency 
    global Subarray_Latency 

    global InterConnection_Latency 
    global BufferLatency 
    global AccumulLatency 

    global GlobalInterConnection_Latency 
    global GlobalBufferLatency 
    global GlobalAccumulLatency 

    global TileInterConnection_Latency 
    global TileBufferLatency 
    global TileAccumulLatency 

    global PEInterConnection_Latency 
    global PEBufferLatency 
    global PEAccumulLatency 

    # Element Latency
    global ShiftAdderLatency 
    global MUXLatency 
    global ADCLatency 
    global DriverLatency 
    global LevelShifterLatency 
    global MemoryArrayLatency 
    global UpdateLatency 
    
    print('-------------------------Latency---------------------------')
    print('Total Chip Latency is {:.9f} s'.format(Total))
    print('Total Chip Latency per image is {:.9f} s'.format(Total/Parameter.DATANUM))
    print('Total Buffer Latency is {:.9f} s'.format(BufferLatency))
    print('Total InterConnection Latency is {:.9f} s'.format(InterConnection_Latency))
    print('Total Accumulator Latency is {:.9f} s'.format(AccumulLatency))
    print('Total SubArray Latency is {:.9f} s'.format(Subarray_Latency))
    print('Total Memory Array Latency is {:.9f} s'.format(MemoryArrayLatency))
    print('Total Driver Latency is {:.9f} s'.format(DriverLatency))
    print('Total Levelshifter Latency is {:.9f} s'.format(LevelShifterLatency))
    print('Total MUX Latency is {:.9f} s'.format(MUXLatency))
    print('Total ADC Latency is {:.9f} s'.format(ADCLatency))
    print('Total shifterAdder Latency is {:.9f} s'.format(ShiftAdderLatency))
    print('Total Weight Update Latency is {:.9f} s'.format(UpdateLatency))
    print('Total Global Buffer Latency is {:.9f} s'.format(GlobalBufferLatency))
    print('Total Tile Buffer Latency is {:.9f} s'.format(TileBufferLatency))
    print('Total PE Buffer Latency is {:.9f} s'.format(PEBufferLatency))
    print('Total Global Accumulator Latency is {:.9f} s'.format(GlobalAccumulLatency))
    print('Total Tile Accumulator Latency is {:.9f} s'.format(TileAccumulLatency))
    print('Total PE Accumulator Latency is {:.9f} s'.format(PEAccumulLatency))
    print('Total Global InterConnection Latency is {:.9f} s'.format(GlobalInterConnection_Latency))
    print('Total Tile InterConnection Latency is {:.9f} s'.format(TileInterConnection_Latency))
    print('Total PE InterConnection Latency is {:.9f} s'.format(PEInterConnection_Latency))

    print('---------------------------------------------------------')