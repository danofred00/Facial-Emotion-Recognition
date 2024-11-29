
const API_URL = 'http://localhost:8000/api/v1';

/**
 * This is responsible to communicate with the backend to 
 * predict the emotion of the user
 */
async function predict_emotion(blob) {

    const formData = new FormData()
    formData.set('image', blob, 'image.png')
    
    const options = {
        body: formData,
        method: 'POST',
        headers: {
            "Accept": "application/json"
        }
    } 

    const response = await fetch(getApiUrl('predict_emotion'), options)
    const json = await response.json()
    return json
}   

/**
 * This function is responsible to fetch the list of avaliables emotions from
 * The remote server
 */
async function getEmtions() {
    const response = await fetch(getApiUrl('get_emotions'))
    return await response.json()
}

/**
 * This return an API endpoint 
 */
function getApiUrl(path) {
    return API_URL + '/' + path
}