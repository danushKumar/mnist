import numpy as np

def avg_dark_intensity(data):
    darkness_intensity = {}
    class_count = {}
    
    for x, y in data:
        
        is_intensity_present = darkness_intensity.get(y[0])
        is_class_present = class_count.get(y[0])
        
        if not is_intensity_present:
            darkness_intensity[y[0]] = 0
            
        if not is_class_present:
            class_count[y[0]] = 0
    
        darkness_intensity[y[0]] += np.sum(x)
        class_count[y[0]] += 1
        
    avg_dark_density = {key: value / class_count[key] 
                            for key, value in darkness_intensity.items()}
    
    return avg_dark_density

def predict(x, avg_intensity):
    difference_cache = {}
    for key, value in avg_intensity.items():
            y = abs(x - value)
            difference_cache[key] = y
    
    
    difference_cache = [(key, value) for key, value in difference_cache.items()]
    difference_cache = sorted(difference_cache, key=lambda x : x[1])
    
    return difference_cache[0][0]

def evaluate(test_data, avg_intensity):
    '''
    hello
    '''
    accuracy = 0

    for x, y in test_data:
        intensity = np.sum(x)
        prediction = predict(intensity, avg_intensity)
        
        if prediction == y:
            accuracy += 1
    
    accuracy = (accuracy / len(test_data)) * 100
    
    return accuracy
