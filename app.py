from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from flask import Flask, request, render_template
app = Flask(__name__)

key = '910d00a739d442ab9e523be23e6ec9e5'
url = 'https://westus.api.cognitive.microsoft.com'
group_id = 'build'

face_client = FaceClient(endpoint=url, credentials=CognitiveServicesCredentials(key))

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        return render_template('train.html')
    elif 'file' not in request.files:
        return 'No file detected'
    image = request.files['file']
    name = request.form['name']

    # Get all the people in the group
    people = face_client.person_group_person.list(group_id)

    # Look to see if the name already exists
    # If not create it
    person = next(p for p in people if p.name.lower() == name.lower())
    if not person:
        person = face_client.person_group_person.create(group_id, name)

    # Add the picture to the person
    face_client.person_group_person.add_face_from_stream(group_id, person.person_id, image)

    # Train the model
    face_client.person_group.train(group_id)
    return 'uploaded picture successfully!!'

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'GET':
        return render_template('detect.html')
    elif 'file' not in request.files:
        return 'No file detected'
    image = request.files['file']

    # Find all the faces in the picture
    faces = face_client.face.detect_with_stream(image)

    # Get just the IDs so we can see who they are
    face_ids = list(map((lambda f: f.face_id), faces))

    # Ask Azure who the faces are
    results = face_client.face.identify(face_ids, 'build')
    names = []
    for result in results:
        # Find the top candidate for each possible face
        top_candidate = sorted(result.candidates, key=lambda c: c.confidence, reverse=True)[0]
        # See who the person is
        person = face_client.person_group_person.get('build', top_candidate.person_id)
        if top_candidate.confidence > .8:
            names.append('I see ' + person.name)
        else:
            names.append('I think I see ' + person.name)
    return render_template('detect.html', names=names)
