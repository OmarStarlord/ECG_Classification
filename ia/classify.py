import tensorflow as tf
import logging
import numpy as np
import os


classifier = None

def generate_model_if_needed(model_name):
  """
  Generate a model if it doesn't exist or can't be loaded
  
  :param model_name: name of the model to generate
  :type model_name: str
  """
  try:
    # Try to import the model generation script
    import generate_models
    
    logging.info(f"Generating {model_name} model with real ECG200 data...")
    
    # Create the graph directory if it doesn't exist
    os.makedirs('./graph', exist_ok=True)
    
    # Load real data and train the model properly
    X_train, y_train, X_test, y_test = generate_models.load_ecg200_data()
    
    if model_name == 'mlp':
      model = generate_models.create_mlp_model()
      generate_models.train_and_save_model(model, model_name, X_train, y_train, epochs=20)  # Reduced epochs for faster generation
    elif model_name == 'cnn':
      model = generate_models.create_cnn_model()
      generate_models.train_and_save_model(model, model_name, X_train, y_train, epochs=30)  # Reduced epochs for faster generation
    elif model_name == 'rnn':
      model = generate_models.create_rnn_model()
      generate_models.train_and_save_model(model, model_name, X_train, y_train, epochs=30)  # Reduced epochs for faster generation
    else:
      raise ValueError(f"Unknown model type: {model_name}")
    
    logging.info(f"Successfully generated and saved {model_name} model")
    
  except Exception as e:
    logging.error(f"Failed to generate {model_name} model: {e}")
    raise

def get_available_models():
  """
  Test all models and return list of ones that can be loaded successfully.
  Generate models if they don't exist.
  
  :return: list of available model names
  :rtype: list
  """
  available_models = []
  models = ['mlp', 'cnn', 'rnn']
  
  for model_name in models:
    model_path = f"./graph/{model_name}_model.h5"
    
    # If model doesn't exist, generate it
    if not os.path.exists(model_path):
      logging.info(f"Model {model_name} not found, generating...")
      try:
        generate_model_if_needed(model_name)
      except Exception as e:
        logging.error(f"Failed to generate {model_name} model: {e}")
        continue
    
    # Try to load the model
    try:
      test_classifier = tf.keras.models.load_model(model_path)
      available_models.append(model_name)
      logging.info(f"Model {model_name} is available")
    except Exception as e:
      logging.warning(f"Model {model_name} failed to load: {e}")
      # Try to regenerate the model
      try:
        logging.info(f"Regenerating {model_name} model...")
        generate_model_if_needed(model_name)
        test_classifier = tf.keras.models.load_model(model_path)
        available_models.append(model_name)
        logging.info(f"Model {model_name} regenerated and is available")
      except Exception as regen_error:
        logging.error(f"Failed to regenerate {model_name} model: {regen_error}")
  
  return available_models

def load_model(model_name: str):
  """
  Load the model with the model name

  :param model_name: the model name
  :type model_name: str
  :raises Exception: if the model is not available
  :raises FileNotFoundError: if the model is not available
  """
  global classifier
  classifier = None
  model_path: str = "./graph/" + model_name + "_model.h5"
  
  if not os.path.exists(model_path):
    logging.error(f"model {model_name} is not available")
    raise FileNotFoundError(f"model {model_name} is not available")
  
  try:
    classifier = tf.keras.models.load_model(model_path)
    logging.info(f"model {model_name} is loaded")
  except Exception as e:
    logging.error(f"Failed to load model {model_name}: {e}")
    raise Exception(f"Failed to load model {model_name}: {e}")
        

def classify_ecg(file_path: str):
  if classifier is None:
    logging.error("Model is not loaded")
    raise Exception("Model is not loaded")
  
  data = np.loadtxt(file_path, delimiter='\t')
  
  expected_length = 96
  if data.ndim == 1:
    data = data[:expected_length]
    data = data.reshape(1, expected_length)
  else:
    data = data[:, :expected_length]
    data = data.reshape(data.shape[0], expected_length)

  # Handle different model input shapes
  if len(classifier.input_shape) == 3:  # CNN or RNN model
    data = data.reshape(data.shape[0], expected_length, 1)

  predictions = classifier.predict(data)
  
  results = {}
  num_digits = len(str(len(predictions)))
  for i, prediction in enumerate(predictions):
    key = f"Data numero {str(i+1).zfill(num_digits)}"
    
    # Handle 2-class softmax output (new models) vs binary output (old models)
    if len(prediction) == 2:
      # 2-class softmax: [P(class_-1), P(class_1)]
      pred_class = 1 if prediction[1] > prediction[0] else -1
    else:
      # Binary sigmoid output
      pred_class = 1 if prediction[0] > 0.5 else -1
    
    results[key] = pred_class

  return results
