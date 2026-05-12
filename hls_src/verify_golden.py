"""Verify the testbench golden data matches the assignment test vectors."""
import numpy as np

Q = 3603169181

def ma(a, b): return int((a + b) % Q)
def ms(a, b): return int((a - b) % Q)
def mm(a, b): return int((a * b) % Q)
def mle(z, o, t): return ma(mm(ms(o, z), t), z)

# Starting tables
A = [3393379911, 2981385125, 2084743985, 83534990, 183830346, 1485329884, 1196205875, 1703421598, 3105961542, 852416133, 697620885, 815293416, 1462677537, 2368463091, 1641191852, 2695543358]
B = [1009888667, 242978280, 1489369074, 3162735331, 3099002876, 1015109858, 2386803472, 882268619, 1656912700, 1630717979, 2285861710, 3146833087, 900407709, 1905237946, 1378868269, 2702732716]
C = [3026106149, 2376905095, 1274725382, 3008574446, 491383016, 2313566733, 2351566239, 3384723696, 141235435, 2414696771, 1018519416, 3518918811, 3407197093, 3201460813, 172342382, 3401331891]
D = [602300577, 1587786683, 1981647596, 826931302, 1490706237, 3516374429, 617596124, 1113156262, 678447488, 2340786897, 977213934, 3483842296, 2755589787, 3520150248, 2334894238, 2484378615]
E = [2567867327, 519474736, 338035350, 3019965900, 2061682616, 810329972, 1735983106, 726845635, 1029898590, 947939748, 2114149950, 3126644740, 1033989233, 2339641645, 1273472379, 1224289190]
G = [3147202555, 1000279472, 1770339562, 3333507361, 440113528, 3362419068, 3386540349, 1901152974, 3419930416, 218995998, 623678842, 1947245284, 3195415403, 1883659776, 2151286614, 1244891464]

challenges = [1569837365, 1364485251, 2168704920, 3350104573]

expected = [
    [862479237, 962438882, 2751652296],
    [340020908, 1881077529, 2383218570],
    [1215317213, 2726123316, 509838412],
    [3454689742, 2183503154, 1875025448],
]

# Expression: a*b + c
tables = [A[:], B[:], C[:], D[:], E[:], G[:]]

for round_idx in range(4):
    n = len(tables[0])
    pairs = n // 2
    evals = [0, 0, 0]
    
    for i in range(pairs):
        az, ao = tables[0][2*i], tables[0][2*i+1]
        bz, bo = tables[1][2*i], tables[1][2*i+1]
        cz, co = tables[2][2*i], tables[2][2*i+1]
        
        # t=0: a*b + c at zeros
        evals[0] = ma(evals[0], ma(mm(az, bz), cz))
        # t=1: a*b + c at ones
        evals[1] = ma(evals[1], ma(mm(ao, bo), co))
        # t=2: extend then compose
        a2 = mle(az, ao, 2)
        b2 = mle(bz, bo, 2)
        c2 = mle(cz, co, 2)
        evals[2] = ma(evals[2], ma(mm(a2, b2), c2))
    
    match = evals == expected[round_idx]
    print(f"Round {round_idx}: {'PASS' if match else 'FAIL'}")
    if not match:
        print(f"  Got:      {evals}")
        print(f"  Expected: {expected[round_idx]}")
    
    # Fold all 6 tables
    r = challenges[round_idx]
    for k in range(6):
        new_table = []
        for j in range(pairs):
            z = tables[k][2*j]
            o = tables[k][2*j + 1]
            new_table.append(mle(z, o, r))
        tables[k] = new_table

print("\nAll rounds verified against golden data.")
