import torch
import Parameter
import math
import Area
import Accuracy
import Latency

# Module Energy
Total = 0
Chip_Energy = 0
Tile_Energy = 0
PE_Energy = 0
Subarray_Energy = 0
InterConnection_E = 0
BufferE = 0
AccumulE = 0

GlobalInterConnection_E = 0
GlobalBufferE = 0
GlobalAccumulE = 0

TileInterConnection_E = 0
TileBufferE = 0
TileAccumulE = 0

PEInterConnection_E = 0
PEBufferE = 0
PEAccumulE = 0

#Element E
ShiftAdderE = 0
MUXE = 0
ADCE = 0
DriverE = 0
LevelShifterE = 0
MemoryArrayE = 0
UpdateE = 0
PoolEner = 0

def Calculate(): #
    global Total
    global Chip_Energy
    global GlobalBuffer
    global GlobalBufferE
    global GlobalAccumulE
    global GlobalInterConnection_E
    global InterConnection_E
    global AccumulE
    global BufferE
    global PoolEner

    TileE = Tile()
    IntCE = InterConnect(Parameter.NumTilePerChip, Parameter.BusToTileT, Area.Tile_height, Area.Tile_width)
    RCANUM = Parameter.ADCPrecision + math.ceil(Parameter.InputPrecesion / Parameter.InputEncoding)
    RCANUM += (Parameter.ColumnsPerWeight-1)
    ChipAccE = Accumulate(Parameter.NumTilePerChip, RCANUM, 0)
    GlobalBuffer = SRAMBuffer(0)

    IntCtotE = 0
    ChipAcctotE = 0
    GlobalBuffertotE = 0
    PoolE = 0
    TotTile = 0
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
                        Parr = min(Parameter.SubRow/Parameter.LayerInChan[i], Parameter.SubCol/Parameter.LayerInChan[i+1])
                    if Parameter.LayerType[i + 1] == 'FC':
                        Parr = min(Parameter.SubRow / Parameter.LayerInChan[i],Parameter.SubCol / Parameter.LayerInChan[i])
                    NumT = (Parameter.Kernel ** 2) * 2 * Parameter.ColumnsPerWeight
                    NumT = math.ceil(NumT/Parr)
                    # Number of PE Elements
                    NumT = math.ceil(1.0 * NumT / Parameter.NumSubperPE)
                    # Number of Tile Elements
                    NumT = math.ceil(1.0 * NumT / Parameter.NumPEperTile)

                TotTile += NumT
                # Global HTree Activation
                NumReadIC = (Parameter.LayerIMG[i]**2) * (Parameter.Kernel**2) * Parameter.InputPrecesion
                NumReadIC *= Parameter.LayerInChan[i]
                if Parameter.LayerType[i+1] == 'Conv':
                    NumReadIC += (Parameter.LayerIMG[i+1]**2) * (Parameter.Kernel**2) * Parameter.InputPrecesion * Parameter.LayerInChan[i+1]
                if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                    NumReadIC += Parameter.LayerInChan[i+1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))

                # Global Accumulator Activation
                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = (Parameter.LayerIMG[i] ** 2) * Parameter.LayerInChan[i+1]

                # Global Buffer Activation
                NumOpBuffer = (Parameter.LayerIMG[i] ** 2) * Parameter.InputPrecesion * (Parameter.Kernel ** 2)
                NumOpBuffer *= Parameter.LayerInChan[i] / Parameter.SRAMCol

                if Parameter.IsPoolLayer[i] == 1:
                    NumReadPool = ((Parameter.LayerIMG[i] / 2) ** 2) * Parameter.LayerInChan[i+1]

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

                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = Parameter.LayerInChan[i+1]

                NumOpBuffer = Parameter.InputPrecesion * Parameter.LayerInChan[i] / Parameter.SRAMCol
                TotTile += NumT
            print(TotTile)
            if Parameter.LayerType[i + 1] == 'END':
                NumOpBuffer += (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) \
                               * Parameter.LayerInChan[i+1] / Parameter.SRAMCol

            IntCtotE += NumReadIC * IntCE
            ChipAcctotE += ChipAccE * NumReadAcc
            GlobalBuffertotE += GlobalBuffer * NumOpBuffer
            PoolE += ChipAccE * NumReadPool

        Total += (IntCtotE + ChipAcctotE + GlobalBuffertotE + PoolE) * Parameter.DATANUM

        if TotTile > Parameter.NumTilePerChip:
            print("The required Tile Number is bigger than you designed!! (Redesign!!!!)")
        Total += TileE
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
                    # Number of Tile Elements
                    NumT = math.ceil(1.0 * NumT / Parameter.NumPEperTile)

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
                    # Number of PE Elements
                    NumT = math.ceil(1.0 * NumT / Parameter.NumSubperPE)
                    # Number of Tile Elements
                    NumT = math.ceil(1.0 * NumT / Parameter.NumPEperTile)

                TotTile += NumT
                # Global HTree Activation
                NumReadIC = (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * Parameter.InputPrecesion
                NumReadIC *= Parameter.LayerInChan[i]
                if Parameter.LayerType[i + 1] == 'Conv':
                    NumReadIC += (Parameter.LayerIMG[i + 1] ** 2) * (Parameter.Kernel ** 2) * Parameter.InputPrecesion * \
                                 Parameter.LayerInChan[i + 1]
                if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                    NumReadIC += Parameter.LayerInChan[i + 1] * (
                                RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))

                # Global Accumulator Activation
                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = (Parameter.LayerIMG[i] ** 2) * Parameter.LayerInChan[i + 1]

                # Global Buffer Activation
                NumOpBuffer = (Parameter.LayerIMG[i] ** 2) * Parameter.InputPrecesion * (Parameter.Kernel ** 2)
                NumOpBuffer *= Parameter.LayerInChan[i] / Parameter.SRAMCol

                if Parameter.IsPoolLayer[i] == 1:
                    NumReadPool = ((Parameter.LayerIMG[i] / 2) ** 2) * Parameter.LayerInChan[i + 1]

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

                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = Parameter.LayerInChan[i + 1]

                NumOpBuffer = Parameter.InputPrecesion * Parameter.LayerInChan[i] / Parameter.SRAMCol
                TotTile += NumT

            if Parameter.LayerType[i + 1] == 'END':
                NumOpBuffer += (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) \
                               * Parameter.LayerInChan[i + 1] / Parameter.SRAMCol

            IntCtotE += NumReadIC * IntCE * (2 + Parameter.DelPrecision/Parameter.InputPrecesion)
            ChipAcctotE += ChipAccE * NumReadAcc * (2 + Parameter.DelPrecision/Parameter.InputPrecesion)
            GlobalBuffertotE += GlobalBuffer * NumOpBuffer * (2 + Parameter.DelPrecision/Parameter.InputPrecesion)
            PoolE += ChipAccE * NumReadPool

        Total += (IntCtotE + ChipAcctotE + GlobalBuffertotE + PoolE) * Parameter.DATANUM
        if TotTile > Parameter.NumTilePerChip:
            print("The required Tile Number is bigger than you designed!! (Redesign!!!!)")
        Total += TileE
    
    PoolEner += PoolE * Parameter.DATANUM
    GlobalBufferE += GlobalBuffertotE * Parameter.DATANUM
    GlobalAccumulE += ChipAcctotE * Parameter.DATANUM
    GlobalInterConnection_E += IntCtotE * Parameter.DATANUM
    BufferE += GlobalBuffertotE * Parameter.DATANUM
    AccumulE += ChipAcctotE * Parameter.DATANUM
    InterConnection_E += IntCtotE * Parameter.DATANUM
    Total += Accuracy.ReadSynEnergy
    Chip_Energy = Total

def Tile():
    global Tile_Energy
    global TileBufferE
    global TileAccumulE
    global TileInterConnection_E
    global InterConnection_E
    global AccumulE
    global BufferE

    # Energy consumption by PEs
    PEE = PE()

    # InterConnect Energu
    IntCE =  InterConnect(Parameter.NumPEperTile, Parameter.BusToPET, Area.PE_height, Area.PE_width)

    # Accumulator Energy
    RCANUM = Parameter.ADCPrecision + math.ceil(Parameter.InputPrecesion / Parameter.InputEncoding)
    RCANUM += (Parameter.ColumnsPerWeight-1)
    TileAccE = Accumulate(Parameter.NumPEperTile, RCANUM, 0)

    # Tile Buffer Energy
    if Parameter.TileBuffer == 'SRAM':
        TileBuffE = SRAMBuffer(0)
        # Not used yet
    if Parameter.TileBuffer == 'DFF':
        TileBuffE = DFF()

    Total = 0
    IntCEnergy = 0
    TileAccEnergy = 0
    TileBuffEnergy = 0
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
                    NumReadIC *= Parameter.LayerInChan[i]
                    if Parameter.LayerType[i + 1] == 'Conv':
                        NumReadIC += (Parameter.LayerIMG[i + 1] ** 2) * (
                                    Parameter.Kernel ** 2) * Parameter.InputPrecesion * Parameter.LayerInChan[i + 1]

                    if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                        NumReadIC += Parameter.LayerInChan[i + 1] * Parameter.InputPrecesion

                # Tile Bus Activation
                if Parameter.BusToPET == 'Bus':
                    NumReadIC = (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * Parameter.InputPrecesion
                    NumReadIC *= Parameter.LayerInChan[i]
                    if Parameter.LayerType[i + 1] == 'Conv':
                        NumReadIC += (Parameter.LayerIMG[i + 1] ** 2) * (Parameter.Kernel ** 2) * (RCANUM +
                            math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))* Parameter.LayerInChan[i + 1]

                    if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                        NumReadIC += Parameter.LayerInChan[i + 1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))

                # Tile Accumulator Activation
                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = (Parameter.LayerIMG[i] ** 2) * Parameter.LayerInChan[i+1]

                # Tile Buffer Activation
                NumOpBuffer = (Parameter.LayerIMG[i] ** 2) * Parameter.InputPrecesion * (Parameter.Kernel ** 2)
                NumOpBuffer *= Parameter.LayerInChan[i] / Parameter.SRAMCol

            if Parameter.LayerType[i] == 'FC':
                # Number of Subarrays
                NumT = math.ceil(Parameter.LayerInChan[i] / Parameter.SubRow) * 2 * Parameter.ColumnsPerWeight
                NumT *= math.ceil(Parameter.LayerInChan[i + 1] / Parameter.SubCol)
                # Number of PE Elements
                NumT = math.ceil(1.0 * NumT / Parameter.NumSubperPE)

                NumReadIC = Parameter.LayerInChan[i] * Parameter.InputPrecesion
                NumReadIC += Parameter.LayerInChan[i + 1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))

                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = Parameter.LayerInChan[i+1]

                NumOpBuffer = Parameter.InputPrecesion * Parameter.LayerInChan[i] / Parameter.SRAMCol

            if Parameter.LayerType[i + 1] == 'Conv':
                NumOpBuffer += (Parameter.LayerIMG[i + 1] ** 2) * (Parameter.Kernel ** 2) * (RCANUM +
                math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) * Parameter.LayerInChan[i + 1] / Parameter.SRAMCol
            if Parameter.LayerType[i + 1] == 'END' or Parameter.LayerType[i + 1] == 'FC':
                NumOpBuffer += (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) \
                               * Parameter.LayerInChan[i+1] / Parameter.SRAMCol
            IntCEnergy += NumReadIC * IntCE
            TileAccEnergy += TileAccE * NumReadAcc
            TileBuffEnergy += TileBuffE * NumOpBuffer

        Total += (IntCEnergy + TileAccEnergy + TileBuffEnergy) * Parameter.DATANUM

        Total += PEE

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
                    NumReadIC *= Parameter.LayerInChan[i]
                    if Parameter.LayerType[i + 1] == 'Conv':
                        NumReadIC += (Parameter.LayerIMG[i + 1] ** 2) * (
                                    Parameter.Kernel ** 2) * Parameter.InputPrecesion * Parameter.LayerInChan[i + 1]

                    if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                        NumReadIC += Parameter.LayerInChan[i + 1] * Parameter.InputPrecesion

                # Tile Bus Activation
                if Parameter.BusToPET == 'Bus':
                    NumReadIC = (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * Parameter.InputPrecesion
                    NumReadIC *= Parameter.LayerInChan[i]
                    if Parameter.LayerType[i + 1] == 'Conv':
                        NumReadIC += (Parameter.LayerIMG[i + 1] ** 2) * (Parameter.Kernel ** 2) * (RCANUM +
                            math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))* Parameter.LayerInChan[i + 1]

                    if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                        NumReadIC += Parameter.LayerInChan[i + 1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))

                # Tile Accumulator Activation
                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = (Parameter.LayerIMG[i] ** 2) * Parameter.LayerInChan[i+1]

                # Tile Buffer Activation
                NumOpBuffer = (Parameter.LayerIMG[i] ** 2) * Parameter.InputPrecesion * (Parameter.Kernel ** 2)
                NumOpBuffer *= Parameter.LayerInChan[i] / Parameter.SRAMCol

            if Parameter.LayerType[i] == 'FC':
                # Number of Subarrays
                NumT = math.ceil(Parameter.LayerInChan[i] / Parameter.SubRow) * 2 * Parameter.ColumnsPerWeight
                NumT *= math.ceil(Parameter.LayerInChan[i + 1] / Parameter.SubCol)
                # Number of PE Elements
                NumT = math.ceil(1.0 * NumT / Parameter.NumSubperPE)

                NumReadIC = Parameter.LayerInChan[i] * Parameter.InputPrecesion
                NumReadIC += Parameter.LayerInChan[i + 1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))

                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = Parameter.LayerInChan[i+1]

                NumOpBuffer = Parameter.InputPrecesion * Parameter.LayerInChan[i] / Parameter.SRAMCol

            if Parameter.LayerType[i + 1] == 'Conv':
                NumOpBuffer += (Parameter.LayerIMG[i + 1] ** 2) * (Parameter.Kernel ** 2) * (RCANUM +
                math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) * Parameter.LayerInChan[i + 1] / Parameter.SRAMCol
            if Parameter.LayerType[i + 1] == 'END' or Parameter.LayerType[i + 1] == 'FC':
                NumOpBuffer += (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) \
                               * Parameter.LayerInChan[i+1] / Parameter.SRAMCol
            IntCEnergy += NumReadIC * IntCE * (2 + Parameter.DelPrecision/Parameter.InputPrecesion)
            TileAccEnergy += TileAccE * NumReadAcc * (2 + Parameter.DelPrecision/Parameter.InputPrecesion)
            TileBuffEnergy += TileBuffE * NumOpBuffer * (2 + Parameter.DelPrecision/Parameter.InputPrecesion)

        Total += (IntCEnergy + TileAccEnergy + TileBuffEnergy) * Parameter.DATANUM

        Total += PEE
        
    TileBufferE += TileBuffEnergy * Parameter.DATANUM
    TileAccumulE += TileAccEnergy * Parameter.DATANUM
    TileInterConnection_E += IntCEnergy * Parameter.DATANUM
    BufferE += TileBuffEnergy * Parameter.DATANUM
    AccumulE += TileAccEnergy * Parameter.DATANUM
    InterConnection_E += IntCEnergy * Parameter.DATANUM
    Tile_Energy = Total
    return Total

def PE():
    global PE_Energy
    global PEBufferE
    global PEAccumulE
    global PEInterConnection_E
    global InterConnection_E
    global AccumulE
    global BufferE

    # Energy consumption by PEs
    SubE = SubArray()
    # InterConnect Energu
    IntCE =  InterConnect(Parameter.NumSubperPE, Parameter.BusToSubT, Area.Subarray_height, Area.Subarray_width)

    # Accumulator Energy
    RCANUM = Parameter.ADCPrecision + math.ceil(Parameter.InputPrecesion / Parameter.InputEncoding)
    RCANUM += (Parameter.ColumnsPerWeight-1)
    PEAccE = Accumulate(Parameter.NumSubperPE, RCANUM ,0)

    # Tile Buffer Energy
    if Parameter.TileBuffer == 'SRAM':
        PEBuffE = SRAMBuffer(0)
        # Not used yet
    if Parameter.TileBuffer == 'DFF':
        PEBuffE = DFF()

    Total = 0
    IntCEnergy = 0
    PEAccEnergy = 0
    PEBuffEnergy = 0
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
                    NumReadIC *= Parameter.LayerInChan[i]
                    if Parameter.LayerType[i + 1] == 'Conv':
                        NumReadIC += (Parameter.LayerIMG[i + 1] ** 2) * (
                                    Parameter.Kernel ** 2) * Parameter.InputPrecesion * Parameter.LayerInChan[i + 1]

                    if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                        NumReadIC += Parameter.LayerInChan[i + 1] * Parameter.InputPrecesion

                # PE Bus Activation
                if Parameter.BusToSubT == 'Bus':
                    NumReadIC = (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * Parameter.InputPrecesion
                    NumReadIC *= Parameter.LayerInChan[i]
                    if Parameter.LayerType[i + 1] == 'Conv':
                        NumReadIC += (Parameter.LayerIMG[i + 1] ** 2) * (
                                Parameter.Kernel ** 2) * Parameter.InputPrecesion * Parameter.LayerInChan[i + 1]

                    if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                        NumReadIC += Parameter.LayerInChan[i + 1] * Parameter.InputPrecesion

                # PE Accumulator Activation
                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = (Parameter.LayerIMG[i] ** 2) * Parameter.LayerInChan[i+1]

                # PE Buffer Activation
                NumOpBuffer = (Parameter.LayerIMG[i] ** 2) * Parameter.InputPrecesion * (Parameter.Kernel ** 2)
                NumOpBuffer *= Parameter.LayerInChan[i] / Parameter.SRAMCol

            if Parameter.LayerType[i] == 'FC':
                # Number of Subarrays
                NumT = math.ceil(Parameter.LayerInChan[i] / Parameter.SubRow) * 2 * Parameter.ColumnsPerWeight
                NumT *= math.ceil(Parameter.LayerInChan[i + 1] / Parameter.SubCol)

                NumReadIC = Parameter.LayerInChan[i] * Parameter.InputPrecesion
                NumReadIC += Parameter.LayerInChan[i + 1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))

                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = Parameter.SubCol / Parameter.ColumnsPerWeight

                NumOpBuffer = Parameter.InputPrecesion * Parameter.LayerInChan[i] / Parameter.SRAMCol

            if Parameter.LayerType[i + 1] == 'Conv':
                NumOpBuffer += (Parameter.LayerIMG[i + 1] ** 2) * (Parameter.Kernel ** 2) * \
                    (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) * \
                               Parameter.LayerInChan[i + 1] / Parameter.SRAMCol
            if Parameter.LayerType[i + 1] == 'END' or Parameter.LayerType[i + 1] == 'FC':
                NumOpBuffer += (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) * \
                               Parameter.LayerInChan[i+1] / Parameter.SRAMCol

            IntCEnergy += NumReadIC * IntCE
            PEAccEnergy += PEAccE * NumReadAcc
            PEBuffEnergy += PEBuffE * NumOpBuffer

        Total += (IntCEnergy + PEAccEnergy + PEBuffEnergy) * Parameter.DATANUM

        Total += SubE

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
                    NumReadIC *= Parameter.LayerInChan[i]
                    if Parameter.LayerType[i + 1] == 'Conv':
                        NumReadIC += (Parameter.LayerIMG[i + 1] ** 2) * (
                                Parameter.Kernel ** 2) * Parameter.InputPrecesion * Parameter.LayerInChan[i + 1]

                    if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                        NumReadIC += Parameter.LayerInChan[i + 1] * Parameter.InputPrecesion

                # PE Bus Activation
                if Parameter.BusToSubT == 'Bus':
                    NumReadIC = (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * Parameter.InputPrecesion
                    NumReadIC *= Parameter.LayerInChan[i]
                    if Parameter.LayerType[i + 1] == 'Conv':
                        NumReadIC += (Parameter.LayerIMG[i + 1] ** 2) * (
                                Parameter.Kernel ** 2) * Parameter.InputPrecesion * Parameter.LayerInChan[i + 1]

                    if Parameter.LayerType[i + 1] == 'FC' or Parameter.LayerType[i + 1] == 'END':
                        NumReadIC += Parameter.LayerInChan[i + 1] * Parameter.InputPrecesion

                # PE Accumulator Activation
                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = (Parameter.LayerIMG[i] ** 2) * Parameter.LayerInChan[i + 1]

                # PE Buffer Activation
                NumOpBuffer = (Parameter.LayerIMG[i] ** 2) * Parameter.InputPrecesion * (Parameter.Kernel ** 2)
                NumOpBuffer *= Parameter.LayerInChan[i] / Parameter.SRAMCol

            if Parameter.LayerType[i] == 'FC':
                # Number of Subarrays
                NumT = math.ceil(Parameter.LayerInChan[i] / Parameter.SubRow) * 2 * Parameter.ColumnsPerWeight
                NumT *= math.ceil(Parameter.LayerInChan[i + 1] / Parameter.SubCol)

                NumReadIC = Parameter.LayerInChan[i] * Parameter.InputPrecesion
                NumReadIC += Parameter.LayerInChan[i + 1] * (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2))))

                if NumT == 1:
                    NumReadAcc = 0
                else:
                    NumReadAcc = Parameter.SubCol / Parameter.ColumnsPerWeight

                NumOpBuffer = Parameter.InputPrecesion * Parameter.LayerInChan[i] / Parameter.SRAMCol

            if Parameter.LayerType[i + 1] == 'Conv':
                NumOpBuffer += (Parameter.LayerIMG[i + 1] ** 2) * (Parameter.Kernel ** 2) * \
                               (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) * \
                               Parameter.LayerInChan[i + 1] / Parameter.SRAMCol
            if Parameter.LayerType[i + 1] == 'END' or Parameter.LayerType[i + 1] == 'FC':
                NumOpBuffer += (RCANUM + math.ceil(math.log2(2 * (Parameter.Kernel ** 2)))) * \
                               Parameter.LayerInChan[i + 1] / Parameter.SRAMCol

            IntCEnergy += NumReadIC * IntCE * (2 + Parameter.DelPrecision/Parameter.InputPrecesion)
            PEAccEnergy += PEAccE * NumReadAcc * (2 + Parameter.DelPrecision/Parameter.InputPrecesion)
            PEBuffEnergy += PEBuffE * NumOpBuffer * (2 + Parameter.DelPrecision/Parameter.InputPrecesion)

        Total += (IntCEnergy + PEAccEnergy + PEBuffEnergy) * Parameter.DATANUM

        Total += SubE
    PEBufferE += PEBuffEnergy * Parameter.DATANUM
    PEAccumulE += PEAccEnergy * Parameter.DATANUM
    PEInterConnection_E += IntCEnergy * Parameter.DATANUM
    BufferE += PEBuffEnergy * Parameter.DATANUM
    AccumulE += PEAccEnergy * Parameter.DATANUM
    InterConnection_E += IntCEnergy * Parameter.DATANUM
        
    PE_Energy = Total
    return Total


def SubArray():
    global Subarray_Energy
    global ADCE
    global MemoryArrayE
    global DriverE
    global LevelShifterE
    global ShiftAdderE
    global MUXE
    global UpdateE

    if Parameter.Mode == 'Inference_Normal':
        ME = MemoryArray()
        DE = Driver(Parameter.SubCol * 0.06e-15)
        LE = Levelshifter(Parameter.SubCol * 0.06e-15)
        if Parameter.ColumnsPerADC > 1:
            MUXE = MUX(Parameter.SubRow * 0.08e-15)
        ADCE = ADC()
        ShAdE = ShiftAdd(0)
        Subarray_E = 0
        TotADCE = 0
        TotLE = 0
        TotDE = 0
        TotShAdE = 0

        for i in range(Parameter.NumLayer):
            if Parameter.LayerType[i] == 'Conv':
                # Driver Operation Number
                InpCy = Parameter.InputPrecesion / Parameter.InputEncoding

                NumOpD = InpCy * (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * Parameter.LayerInChan[i] * 2

                # Level Shifter Operation Number
                NumOpL = 0  # This is Inferencing Moder

                # ADC Operation Numbers
                NumOpADC =  InpCy * Parameter.ColumnsPerWeight * Parameter.LayerInChan[i + 1]
                NumOpADC *= (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * 2

                # Shift Adder Operation Numbers
                if Parameter.InputPrecesion != Parameter.InputEncoding or Parameter.ColumnsPerWeight != 0:
                    NumOpShAd = InpCy * Parameter.ColumnsPerWeight * Parameter.LayerInChan[i + 1] * 2
                    NumOpShAd *= (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2)

            if Parameter.LayerType[i] == 'FC':
                # Driver Operation Number
                InpCy = Parameter.InputPrecesion / Parameter.InputEncoding

                NumOpD = InpCy * Parameter.LayerInChan[i] * math.ceil(Parameter.LayerInChan[i + 1] / Parameter.SubCol)
                NumOpD *= 2

                # Level Shifter Operation Number
                NumOpL = 0  # This is Inferencing Moder

                # ADC Operation Numbers
                NumOpADC =  InpCy * Parameter.ColumnsPerWeight * Parameter.LayerInChan[i + 1]
                NumOpADC *= math.ceil(Parameter.LayerInChan[i + 1] * 1.0 / Parameter.SubRow) * 2

                # Shift Adder Operation Numbers
                if Parameter.InputPrecesion != Parameter.InputEncoding or Parameter.ColumnsPerWeight != 0:
                    NumOpShAd = InpCy * Parameter.ColumnsPerWeight * Parameter.LayerInChan[i + 1] * 2
                    NumOpShAd *= math.ceil(Parameter.LayerInChan[i + 1] / Parameter.SubRow)

            Subarray_E += ((DE * NumOpD) + (LE * NumOpL) + (ADCE * NumOpADC) + (ShAdE * NumOpShAd)) * Parameter.DATANUM
            Subarray_E += Accuracy.ReadSynEnergy
            TotLE += (LE * NumOpL) * Parameter.DATANUM
            TotADCE += (ADCE * NumOpADC) * Parameter.DATANUM
            TotDE += (DE * NumOpD) * Parameter.DATANUM
            TotShAdE += (ShAdE * NumOpShAd) * Parameter.DATANUM

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
        ME = MemoryArray()
        DE = Driver(Parameter.SubCol * 0.06e-15)
        LE = Levelshifter(Parameter.SubCol * 0.06e-15)
        if Parameter.ColumnsPerADC > 1:
            MUXE = MUX(Parameter.SubRow * 0.08e-15)
        ADCE = ADC()
        ShAdE = ShiftAdd(0)
        Subarray_E = 0
        TotADCE = 0
        TotLE = 0
        TotDE = 0
        TotShAdE = 0

        for i in range(Parameter.NumLayer):
            if Parameter.LayerType[i] == 'Conv':
                # Driver Operation Number
                InpCy = Parameter.InputPrecesion / Parameter.InputEncoding
                InpCy2 = Parameter.DelPrecision / Parameter.DelEn

                NumOpD = InpCy * (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * Parameter.LayerInChan[i] * 2
                if Parameter.LayerType[i + 1] == 'Conv':
                    NumOpD += InpCy2 * (Parameter.LayerIMG[i+1] ** 2) * (Parameter.Kernel ** 2) * Parameter.LayerInChan[i+1] * 2
                if Parameter.LayerType[i + 1] == 'FC':
                    NumOpD += InpCy2 * (Parameter.LayerIMG[i+1])

                # Level Shifter Operation Number
                NumOpL = 0  # This is Considered in Parallel Weight Update Part

                # ADC Operation Numbers
                NumOpADC =  InpCy * Parameter.ColumnsPerWeight * Parameter.LayerInChan[i + 1]
                NumOpADC *= (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * 2

                if Parameter.LayerType[i + 1] == 'Conv':
                    NumOpADC += InpCy2 * Parameter.LayerInChan[i + 1] * (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2) * 2
                if Parameter.LayerType[i + 1] == 'FC':
                    NumOpADC += InpCy2 * (Parameter.LayerIMG[i+1]) * 2

                # Shift Adder Operation Numbers
                if Parameter.InputPrecesion != Parameter.InputEncoding or Parameter.ColumnsPerWeight != 0:
                    NumOpShAd = InpCy * Parameter.ColumnsPerWeight * Parameter.LayerInChan[i + 1] * 2
                    NumOpShAd *= (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2)

                if Parameter.DelPrecision != Parameter.DelEn:
                    if Parameter.LayerType[i + 1] == 'Conv':
                        NumOpShAd += InpCy * Parameter.ColumnsPerWeight * Parameter.LayerInChan[i + 1] * 2 \
                                     * (Parameter.LayerIMG[i] ** 2) * (Parameter.Kernel ** 2)
                    if Parameter.LayerType[i + 1] == 'FC':
                        NumOpShAd += InpCy2 * Parameter.LayerInChan[i + 1] * 2

            if Parameter.LayerType[i] == 'FC':
                # Driver Operation Number
                InpCy = Parameter.InputPrecesion / Parameter.InputEncoding
                InpCy2 = Parameter.DelPrecision / Parameter.DelEn

                NumOpD = InpCy * Parameter.LayerInChan[i] * math.ceil(Parameter.LayerInChan[i + 1] / Parameter.SubCol)
                NumOpD *= 2
                NumOpD += InpCy2 * (Parameter.LayerIMG[i + 1])

                # Level Shifter Operation Number
                NumOpL = 0  # This is Inferencing Moder

                # ADC Operation Numbers
                NumOpADC =  InpCy * Parameter.ColumnsPerWeight * Parameter.LayerInChan[i + 1]
                NumOpADC *= math.ceil(Parameter.LayerInChan[i + 1] * 1.0 / Parameter.SubRow) * 2
                NumOpADC += InpCy2 * (Parameter.LayerIMG[i + 1]) * 2

                # Shift Adder Operation Numbers
                if Parameter.InputPrecesion != Parameter.InputEncoding or Parameter.ColumnsPerWeight != 0:
                    NumOpShAd = InpCy * Parameter.ColumnsPerWeight * Parameter.LayerInChan[i + 1] * 2
                    NumOpShAd *= math.ceil(Parameter.LayerInChan[i + 1] / Parameter.SubRow)

                if Parameter.DelPrecision != Parameter.DelEn:
                    NumOpShAd += InpCy2 * Parameter.LayerInChan[i + 1] * 2

            Subarray_E += ((DE * NumOpD) + (LE * NumOpL) + (ADCE * NumOpADC) + (ShAdE * NumOpShAd)) * Parameter.DATANUM
            TotLE += (LE * NumOpL) * Parameter.DATANUM
            TotADCE += (ADCE * NumOpADC) * Parameter.DATANUM
            TotDE += (DE * NumOpD) * Parameter.DATANUM
            TotShAdE += (ShAdE * NumOpShAd) * Parameter.DATANUM
        Subarray_E += Accuracy.UpdateCapEnergy + Accuracy.UpdateCurrEnergy + Accuracy.ReadSynEnergy
    ADCE += TotADCE
    DriverE += TotDE
    MemoryArrayE += Accuracy.ReadSynEnergy
    ShAdE += TotShAdE
    LevelShifterE += Accuracy.UpdateCapEnergy
    UpdateE += Accuracy.UpdateCurrEnergy
    
    Subarray_Energy = Subarray_E
    return  Subarray_E

def Driver(CapOut):
    # Single Driver Latency
    if Parameter.HighVoltage > Parameter.VDD:  # Use an AND gate besides Levelshifter for Driver
        # DFF Energy
        CapIn = Parameter.GateCapN + Parameter.GateCapP
        EnergyD = DFF(CapIn)

        # Using CV^2
        VDDVDD = Parameter.VDD * Parameter.VDD
        Cap1 = Parameter.JuncCapN + Parameter.JuncCapP * 2 + 2 * Parameter.GateCapP + Parameter.GateCapN
        Energy1 = Cap1 * VDDVDD

        # Using CV^2
        Cap2 = Parameter.JuncCapN + Parameter.JuncCapP * 2 + Parameter.SubCol * 0.06e-15 + CapOut
        Energy2 = Cap2 * VDDVDD

        Energy = EnergyD + Energy1 + Energy2

    if Parameter.HighVoltage <= Parameter.VDD:       # In this case, we can adopt WLDriver Suggested by S.Yu
        # DFF Energy
        CapIn = (Parameter.GateCapN * (1 + 2) + Parameter.GateCapP * 2)
        EnergyD = DFF(CapIn)

        # Driver Energy
        VDDVDD = Parameter.VDD * Parameter.VDD
        CArray = Parameter.SubCol * 0.06e-15
        Energy1 = (CArray + 8*Parameter.JuncCapP + 4*Parameter.JuncCapN) * VDDVDD
        Energy = EnergyD + Energy1

    return Energy

def PWM():
    # Will Be added Later
    return 0

def Levelshifter(CapOut):
    # Single Levelshifter Energy

    VDDVDD = Parameter.VDD*Parameter.VDD
    Cap1 = (Parameter.JuncCapP * 20 + Parameter.JuncCapN * 15) + Parameter.GateCapN * 32 * 6
    Cap1 += (Parameter.JuncCapP * 20 + Parameter.GateCapN * 15)
    Energy1 = Cap1 * VDDVDD

    HVDDHVDD = Parameter.HighVoltage * Parameter.HighVoltage
    L = math.sqrt(Parameter.HighVoltage/3.3) * (270 / Parameter.Tech)
    Cap2 = ((Parameter.JuncCapP *10+Parameter.JuncCapN*32)*2 + Parameter.GateCapN * 146 * L + CapOut)
    Cap2 += 6e-17 * Parameter.SubCol + Parameter.JuncCapP * 64 + Parameter.JuncCapN * 82
    Energy2 = Cap2 * HVDDHVDD

    Energy = Energy1 + Energy2
    return Energy

def MemoryArray():
    #Latency By Array is considered in the Network Processing
    return 0

def MUX(CapOut):
    # MUX doesn't take Energy
    return 0

def ADC():
    # Single ADC Energy
    #Calculating Unit ADC Energy
    if Parameter.ADCType == 'SAR':  # Here we adopted Formula From S.Yu
        if Parameter.Tech == 45:
            Energy = (2.1843 * Parameter.ADCPrecision + 11.931) * 1e-6
            #Energy += 0.097754*math.exp(-2.313*math.log10(Parameter.RA * Parameter.SubRow))
            # We Use Current Mirror & Edge Cap = Infinite Column Resistance Seen From ADC

        Energy *= (Parameter.ADCPrecision + 1) * Parameter.ADCCycle

    if Parameter.ADCType == 'Flash':
        Energy = 0     # Here will be added later

    return Energy

def ShiftAdd(CapOut):
    # Single DFF Energy
    numDFF = (Parameter.ADCPrecision + Parameter.InputEncoding)
    numAdder = Parameter.ADCPrecision

    EnergyD = numDFF * DFF(Parameter.GateCapN * 2 + Parameter.GateCapP * 2)

    # Single Full Adder Energy
    EnergyA = Accumulate(2, numAdder, 0)

    # Single Shift Adder Energy Consumption
    Energy = EnergyD + EnergyA

    return Energy

def Accumulate(NumtoAcc, RCANUM, CapOut):
    # Single Full Adder Latency
    # NumtoAcc (Number of variables)
    # We will use approximated Activity Factor

    VDDVDD = Parameter.VDD * Parameter.VDD
    # First Node
    Act1 = 0.5 * 0.5
    Cap1 = (Parameter.GateCapN * 2 + Parameter.GateCapP * 2)
    Energy1 = Cap1 * Act1 * VDDVDD

    # Second Node
    Act2 = 0.5 * 0.5
    Cap2 = (Parameter.GateCapN * 2 + Parameter.GateCapP * 2)
    Energy2 = Cap2 * Act2 * VDDVDD

    # Third Node
    Act3 = 0.5 * 0.5
    Cap3 = (Parameter.GateCapN * 1 + Parameter.GateCapP * 1)
    Energy3 = Cap3 * Act3 * VDDVDD

    # 4th Node
    Act4 = 0.25 * 0.75
    Cap4 = (Parameter.GateCapN * 3 + Parameter.GateCapP * 3 + Parameter.JuncCapP * 2 + Parameter.JuncCapN * 1)
    Energy4 = Cap4 * Act4 * VDDVDD

    # 5th Node
    Act5 = 0.625 * 0.325
    Cap5 = (Parameter.GateCapN * 1 + Parameter.GateCapP * 1 + Parameter.JuncCapP * 2 + Parameter.JuncCapN * 1)
    Energy5 = Cap5 * Act5 * VDDVDD

    # 6th Node
    Act6 = 0.625 * 0.325
    Cap6 = (Parameter.GateCapN * 1 + Parameter.GateCapP * 1 + Parameter.JuncCapP * 2 + Parameter.JuncCapN * 1)
    Energy6 = Cap6 * Act6 * VDDVDD

    # 7th Node
    Act7 = 0.625 * 0.375
    Cap7 = (Parameter.GateCapN * 2 + Parameter.GateCapP * 2 + Parameter.JuncCapP * 2 + Parameter.JuncCapN * 1)
    Energy7 = Cap7 * Act7 * VDDVDD

    # 8th Node
    Act8 = 0.7 * 0.3
    Cap8 = (Parameter.GateCapN * 3 + Parameter.GateCapP * 3 + Parameter.JuncCapP * 2 + Parameter.JuncCapN * 1)
    Energy8 = Cap8 * Act8 * VDDVDD

    # 9th Node
    Act9 = 0.55 * 0.45
    Cap9 = (Parameter.GateCapN * 1 + Parameter.GateCapP * 1 + Parameter.JuncCapP * 2 + Parameter.JuncCapN * 1)
    Energy9 = Cap9 * Act9 * VDDVDD

    # 10th Node
    Act10 = 0.625 * 0.375
    Cap10 = (Parameter.GateCapN * 1 + Parameter.GateCapP * 1 + Parameter.JuncCapP * 2 + Parameter.JuncCapN * 1)
    Energy10 = Cap10 * Act10 * VDDVDD

    # 11th Node
    Act11 = 0.625 * 0.375
    Cap11 = (CapOut + Parameter.JuncCapP * 2 + Parameter.JuncCapN * 1)
    Energy11 = Cap11 * Act11 * VDDVDD

    # 12th Node
    Act12 = 0.5 * 0.5
    Cap12 = (Parameter.JuncCapP * 2 + Parameter.JuncCapN * 1)
    Energy12 = Cap12 * Act12 * VDDVDD

    # Single Ripple Carray Adder Latency

    Energy = Energy1 + Energy2 + Energy3 + Energy4 + Energy5 + Energy6 + Energy7 + Energy8 + Energy9 + Energy10
    Energy += Energy11 + Energy12
    Energy *= RCANUM
    Energy *= math.ceil(math.log2(NumtoAcc))

    return Energy

def SRAMBuffer(CapOut):
    #SRAM Energy Per ROW Read & Write

    # Single Decoder Energy (Per Row)

    VDDVDD = Parameter.VDD*Parameter.VDD

    NumInNOR = math.ceil(math.log2(Parameter.SRAMRow) / 2.0)
    NumPreDEC = math.floor(math.log2(Parameter.SRAMRow) / 2.0)

    # Predecoder
    Act = 0.5 * 0.5
    Cap1 = (Parameter.JuncCapP * 2 + Parameter.JuncCapN + Parameter.GateCapN * 3 + Parameter.GateCapP * 4)
    Energy1 =  Cap1 * VDDVDD * Act * 2      # multiply 2 , its 2:4 Predecoder and its first node

    Act2 = 0.25 * 0.75
    Cap2 = (Parameter.JuncCapN + Parameter.JuncCapP * 2 + Parameter.GateCapN + Parameter.GateCapP * 2 * NumInNOR)
    Energy2 = Cap2 * VDDVDD * Act2 * 4      # multiply 4 , its 2:4 Predecoder and its second node

    Act3 = (1 * 127.0)/(128 * 128)
    Cap3 = Parameter.SRAMCol * 8e-17 / 2 + Parameter.JuncCapN * 1 + Parameter.JuncCapP * 2 * NumInNOR
    Energy3 = Parameter.SRAMRow * Cap3 * VDDVDD * Act3

    EnergyDEC = (Energy1 + Energy2) * NumPreDEC + Energy3

    # Cell Precharge Latency

    CapP = Parameter.SRAMRow * Parameter.JuncCapN
    EnergyP = CapP * VDDVDD * Parameter.SRAMCol * 2

    # Single Cell Read Latency
    ReadEnergy = EnergyDEC + EnergyP + 0.8e-15 * Parameter.SRAMCol

    # Single Cell Write Latency
    WriteEnergy = EnergyDEC + EnergyP + 3.05e-15 * Parameter.SRAMCol

    Energy = ReadEnergy + WriteEnergy

    return Energy

def DFF(CapOut):
    # Single DFF Energy
    # Switching probability = 0.5
    Act = 0.5 * 0.5
    VDDVDD = Parameter.VDD * Parameter.VDD

    # First Node
    Cap1 = (Parameter.GateCapN + Parameter.GateCapP)
    Energy1 = Act * Cap1 * VDDVDD * 2   #DFF has two latches

    # Second Node (CLK NODE)
    Cap2 = (Parameter.GateCapN * 5 + Parameter.GateCapP * 6 + Parameter.JuncCapP * 2 + Parameter.JuncCapN)
    Energy2 = 1 * Cap2 * VDDVDD

    # Third Node
    Cap3 = (Parameter.GateCapN * 2 + Parameter.GateCapP * 2 + Parameter.JuncCapN + Parameter.JuncCapP * 2)
    Energy3 = Act * Cap3 * VDDVDD * 2

    # 4th Node
    Cap4 = (Parameter.GateCapN * 1 + Parameter.GateCapP * 1 + Parameter.JuncCapN + Parameter.JuncCapP * 2)
    Energy4 = Act * Cap4 * VDDVDD * 2

    # 5th Node
    Cap5 = (Parameter.GateCapN * 1 + Parameter.GateCapP * 1 + Parameter.JuncCapN + Parameter.JuncCapP * 2 + CapOut)
    Energy5 = Act * Cap5 * VDDVDD * 2

    # 6th Node
    Cap6 = (Parameter.GateCapN * 1 + Parameter.GateCapP * 1 + Parameter.JuncCapN + Parameter.JuncCapP * 2)
    Energy6 = Act * Cap6 * VDDVDD * 2

    Energy = Energy1 + Energy2 + Energy3 + Energy4 + Energy5 + Energy6

    return Energy

def InterConnect(NumElements, Type, Height, Width):
    VDDVDD = Parameter.VDD * Parameter.VDD
    if Type == 'HTree':
        Energy = 0
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
            UnitEnergy = (MININVCAPTO + Parameter.UnitWiCap * MINDIST) * VDDVDD / MINDIST * 0.25
        if numREP < 1:
            UnitEnergy = Parameter.UnitWiCap * VDDVDD * 0.1
        Energy += 1e-3 * WireLenH * UnitEnergy

        # Repeats with Stages
        for i in range(0, int((numStage) / 2)):
            WireLenW /= 2
            numREP = 1e-3 * WireLenW / MINDIST
            if numREP >= 1:
                UnitEnergy = (MININVCAPTO + Parameter.UnitWiCap * MINDIST) * VDDVDD / MINDIST * 0.25
            if numREP < 1:
                UnitEnergy = Parameter.UnitWiCap * VDDVDD * 0.25
            Energy += 1e-3 * WireLenW * UnitEnergy

            WireLenH /= 2
            numREP = 1e-3 * WireLenH / MINDIST
            if numREP >= 1:
                UnitEnergy = (MININVCAPTO + Parameter.UnitWiCap * MINDIST) * VDDVDD / MINDIST * 0.25
            if numREP < 1:
                UnitEnergy = Parameter.UnitWiCap * VDDVDD * 0.25
            Energy += 1e-3 * WireLenH * UnitEnergy

    if Type == 'Bus':
        Energy = 0
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
            UnitEnergy = (MININVCAPTO + Parameter.UnitWiCap * MINDIST) * VDDVDD / MINDIST * 0.25
        if numREP < 1:
            UnitEnergy = Parameter.UnitWiCap * VDDVDD * 0.25
        Energy += 1e-3 * WireLength * UnitEnergy
    return Energy

def Print(): # Print Areas
    global Total
    global Chip_Energy
    global Tile_Energy
    global PE_Energy
    global Subarray_Energy

    global InterConnection_E 
    global BufferE 
    global AccumulE 

    global GlobalInterConnection_E 
    global GlobalBufferE 
    global GlobalAccumulE 

    global TileInterConnection_E 
    global TileBufferE 
    global TileAccumulE 

    global PEInterConnection_E 
    global PEBufferE 
    global PEAccumulE 

    # Element E
    global ShiftAdderE 
    global MUXE 
    global ADCE 
    global DriverE 
    global LevelShifterE 
    global MemoryArrayE 
    global UpdateE 
    global PoolEner

    NUMOP = 0

    for i in range(Parameter.NumLayer):
        if Parameter.LayerType[i] == 'Conv':
            IMGS =  Parameter.LayerIMG[i] ** 2
            if Parameter.LayerType[i+1] == 'Conv':
                NUMOP += 2 * Parameter.LayerInChan[i] * Parameter.LayerInChan[i+1] * (Parameter.Kernel ** 2) * IMGS * Parameter.DATANUM
            if Parameter.LayerType[i+1] == 'FC':
                NUMOP += 2 * Parameter.LayerInChan[i] * Parameter.LayerInChan[i] * (Parameter.Kernel ** 2) * IMGS * Parameter.DATANUM


        if Parameter.LayerType[i] == 'FC':
            NUMOP += 2 * Parameter.LayerInChan[i] * Parameter.LayerInChan[i+1] * Parameter.DATANUM
    if Parameter.Mode == 'Inference_Normal':

        NUMOP *= 1

    if Parameter.Mode == 'Onchip_Parallel':
        NUMOP *= 3
        NUMOP -= 2 * ((Parameter.IMGSIZE ** 2 )* Parameter.LayerInChan[0] * Parameter.LayerInChan[1] * (Parameter.Kernel ** 2)) * Parameter.DATANUM


    print('-------------------------Energy---------------------------')
    print('Total Chip Energy is {:.9f} J'.format(Total))
    print('Total Chip Energy per image is {:.9f} J'.format(Total/Parameter.DATANUM))
    print('Total Pooling Energy is {:.9f} J'.format(PoolEner))
    print('Total Buffer Energy is {:.9f} J'.format(BufferE))
    print('Total InterConnection Energy is {:.9f} J'.format(InterConnection_E))
    print('Total Accumulator Energy is {:.9f} J'.format(AccumulE))
    print('Total SubArray Energy is {:.9f} J'.format(Subarray_Energy))
    print('Total Memory Array Energy is {:.9f} J'.format(MemoryArrayE))
    print('Total Driver Energy is {:.9f} J'.format(DriverE))
    print('Total Levelshifter Energy is {:.9f} J'.format(LevelShifterE))
    print('Total ADC Energy is {:.9f} J'.format(ADCE))
    print('Total shifterAdder Energy is {:.9f} J'.format(ShiftAdderE))
    print('Total Weight Update Energy is {:.9f} J'.format(UpdateE))
    print('Total Global Buffer Energy is {:.9f} J'.format(GlobalBufferE))
    print('Total Tile Buffer Energy is {:.9f} J'.format(TileBufferE))
    print('Total PE Buffer Energy is {:.9f} J'.format(PEBufferE))
    print('Total Global Accumulator Energy is {:.9f} J'.format(GlobalAccumulE))
    print('Total Tile Accumulator Energy is {:.9f} J'.format(TileAccumulE))
    print('Total PE Accumulator Energy is {:.9f} J'.format(PEAccumulE))
    print('Total Global InterConnection Energy is {:.9f} J'.format(GlobalInterConnection_E))
    print('Total Tile InterConnection Energy is {:.9f} J'.format(TileInterConnection_E))
    print('Total PE InterConnection Energy is {:.9f} J'.format(PEInterConnection_E))
    print('---------------------------------------------------------')
    print('----------------------Performance------------------------')
    print('TOPS is {:.9f}'.format(1e-12 * NUMOP/Latency.Total))
    print('TOPS/W is {:.9f}'.format(1e-12 * NUMOP/Total))
    print('TOPS/mm2 is {:.9f}'.format(1e-12 * NUMOP/Latency.Total/Area.Total))
    print('---------------------------------------------------------')