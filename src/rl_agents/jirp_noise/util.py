def run_eqv_noise(epsilon, output1, output2):
    """
    Returns True if all outputs are within epsilon of each other (output1 is a noise-distorted output2, eg.)
    """
    if len(output1) != len(output2):
        return False
    for i in range(0, len(output1)):
        if abs(output1[i] - output2[i]) > epsilon:
            return False
    return True
