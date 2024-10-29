/**
 * This is a simple script that fetches data from the API and displays it on the page.
 */

const API_URL = 'http://localhost:8000/api/v1';

document.addEventListener('DOMContentLoaded', () => {

    const button = document.querySelector('#myButton');
    const view = document.querySelector('#content');

    button.addEventListener('click', async () => {
        const response = await fetch(`${API_URL}`);
        const json = await response.json();
        const text = JSON.stringify(json, null, 2);

        view.innerHTML = text;
    });
})

