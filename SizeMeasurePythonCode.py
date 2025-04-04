from flask import Flask, flash, request

from size_calculator_utils import calculate, file_storage_to_cv2

app = Flask(__name__)

@app.route('/calculate_measures', methods=['POST'])
def calculate_measures():
    try:
        if request.method == 'POST':
            if "height" not in request.form:
                flash('No height')
                return {"error": "Height is required"}
            if 'front' not in request.files:
                flash('No file part')
                return {"error": "Front image is required"}
            if 'side' not in request.files:
                flash('No file part')   
                return {"error": "Side image is required"}


            Height = (int)(request.form.get('height'))
            front = request.files['front']
            side = request.files['side']
            front_image = file_storage_to_cv2(front)
            side_image = file_storage_to_cv2(side)
            
            # front.save(front_path)
            # side.save(side_path)
            
            if front.filename == '':
                flash('No selected file')
            if side.filename == '':
                flash('No selected file')
        else:
            print("Shouldn't use GET method")
            return

        image_base_64, body_parts = calculate(front_image, side_image, Height)
        return {'image': image_base_64, "body_parts": body_parts}
       
    except Exception as e:
        print(e)
        return {'error': f"Error reason: {e}"}

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8000)