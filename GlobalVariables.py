class GlobalVariables :

    #options for running different experiments
    use_samples = 0
    use_pitch = 0
    use_spectrogram = 1
    use_raw_data = 0

    #Grid Size
    nRow = 2
    nCol = 2

    use_dense=0
    use_CNN_1D=0
    use_CNN_2D=1

    #parameters
    sample_state_size = 100
    pitch_state_size= 114

    pitch_length=1
    spectrogram_length=129


    spectrogram_state_size= 259
    raw_data_state_size= 100
    action_size = 4
    batch_size = 32
    Number_of_episodes=50
    timesteps=(nRow+nCol+nRow)
    how_many_times = 5 #How many times to run the same experiment

