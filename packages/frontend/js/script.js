
const EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

document.addEventListener('DOMContentLoaded', async () => {

    // Récupérer les éléments HTML
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const fileInput = document.getElementById('fileInput');
    const uploadButton = document.getElementById('uploadButton');
    const captureButton = document.getElementById('captureButton');
    const capturedImage = document.getElementById("captured-image")
    //
    const width = 480
    const height = 480

    // Accéder à la caméra avant de l'utilisateur
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } }).then(stream => {
        video.srcObject = stream;
    }).catch(error => {
        console.error("Erreur d'accès à la caméra:", error);
    });

    // Gérer le clic sur le bouton d'importation de l'image
    uploadButton.addEventListener('click', async (e) => {

        const errorMessage = document.querySelector("#error-image-message")
        const file = fileInput.files[0];
        // display error message box
        if (!file) {
            errorMessage.style.display = "inline"
            return;
        } else {
            errorMessage.style.display = "none"
        }

        //
        capturedImage.src = URL.createObjectURL(file)

        // perform an api request to retrieve the emotion
        try {
            const response = await predict_emotion(file)
            showDataToView(response.data)
        } catch (err) {
            console.error(err)
        }

    });

    // Gérer le clic sur le bouton de capture d'image avec la webcam
    captureButton.addEventListener('click', async () => {

        canvas.setAttribute("width", width)
        canvas.setAttribute("height", height)
        ctx = canvas.getContext('2d')

        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(async (blob) => {
            capturedImage.src = URL.createObjectURL(blob)
            // perform an api request to retrieve the emotion
            try {
                const response = await predict_emotion(blob)
                showDataToView(response.data)
            } catch (err) {
                console.error(err)
            }
        })

    });

    function showDataToView(data) {
        const titles = document.querySelectorAll('.emotions-container h5')
        const progressBars = document.querySelectorAll('.emotions-container .progress-bar')

        data.forEach((value, index) => {
            // this is because each value is in for of 1/100
            const percent = `${Math.round(value * 100 * 100) / 100}%`

            // update titles
            titles[index].textContent = `${EMOTIONS[index]} - ${percent}`

            // update progressBars
            progressBars[index].style.width = percent
        })
    }

})

