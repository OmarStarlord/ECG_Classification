import classify
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

UPLOAD_FOLDER = './downloads'
ALLOWED_EXTENSIONS = {'tsv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.abspath(UPLOAD_FOLDER)

@app.route ('/config', methods=['POST'])
def doConfiguration():
    """
    Route to configure the classifier

    :return: the configuration result
    """
    logging.info("Configuring")
    model_name = "mlp"
    if 'algo' in request.args:
        model_name = request.args['algo']
    
    # Check if model is available
    available_models = classify.get_available_models()
    if model_name not in available_models:
        if available_models:
            error_msg = f"Model '{model_name}' is not available. Available models: {', '.join(available_models)}"
        else:
            error_msg = "No models are currently available"
        logging.error(error_msg)
        return { "status" : "error", "data": error_msg }, 400
    
    try:
        classify.load_model(model_name)
        logging.info("done")
        return { "status" : "ok", "data": f"configuration is done with {model_name} model" }
    except Exception as e:
        error_msg = f"Failed to load model {model_name}: {str(e)}"
        logging.error(error_msg)
        return { "status" : "error", "data": error_msg }, 500
   
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route ('/classify', methods=['POST'])
def doClassification():
    logging.info("Classifying")
    if 'datafile' not in request.files:
        logging.error("No file part")
        return("bad url", 400)
    file = request.files['datafile']
    if not file:
        logging.error("Not file")
        return ("file is wrong", 400)
    if file.filename == '':
        logging.error("No selected file")
        return ("file is empty", 400)
    if not allowed_file(file.filename):
        logging.error("Not allowed file. Must be .tsv")
        return ("file not allowed. Must be .tsv", 400)
    filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filename)
    result = classify.classify_ecg(filename)
    logging.info(result)
    logging.info("done")
    return { "status" : "ok", "data": result, "filename": filename }
  
@app.route ('/', methods=['GET'])
def doHome():
    """
    Route to the home page

    :return: the home page
    """
    # Get available models
    available_models = classify.get_available_models()
    
    # Generate options for available models
    model_options = ""
    for model in available_models:
        selected = "selected" if model == "mlp" else ""
        model_options += f'                <option value="{model}" {selected}>{model.upper()}</option>\n'
    
    # If no models available, show message
    if not available_models:
        model_options = '                <option value="" disabled>No models available</option>\n'
    
    return f'''
    <!doctype html>
    <html>
        <head>
            <title>Classifying ECG</title>
                <meta charset="UTF-8" />
        </head>
        <body>
            <h1>Detection ECG</h1>
            <h2>Choix du modèle</h2>
            <form method=post action="/config" enctype="application/x-www-form-urlencoded" name="config">
            <label for="algo">Algo name ?</label>
            <select name="algo" id="algo">
{model_options}            </select>
            <input type="submit" value="Config">
            </form>
            <h2>Classify ECG</h2>
            <form method="post" action="/classify" enctype="multipart/form-data" name="classify">
                <input type="file" name="datafile" accept=".tsv">
                <input type="submit" value="Classify">
            </form>
        </body>
    </html>
    '''

if __name__ == '__main__' :
    print ("starting")
    app.run(host='0.0.0.0', port=80, debug=True, use_reloader=False)
    print ("done")
