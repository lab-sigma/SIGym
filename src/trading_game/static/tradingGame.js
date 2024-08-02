document.addEventListener("DOMContentLoaded", () => {
    function getElementValue(elementId) {
        var element = document.getElementById(elementId);
        var elementValue = element.innerHTML;

        fetch('/get_html_value', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                'element_id': elementId,
                'element_value': elementValue
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Element ID:', data.element_id);
            console.log('Element Value:', data.element_value);
        })
        .catch(error => console.error('Error:', error));
    }

    // Autoupdate overall deck size when related fields are changed
    document.getElementById('num-suits-input').addEventListener('input', updateDeckSize)
    document.getElementById('cards-per-suit-input').addEventListener('input', updateDeckSize)

    // Update deck size field
    function updateDeckSize() {
        let deckSizeEl = document.getElementById("deck-size-input")
        let numSuits = document.getElementById("num-suits-input").value
        let cardsPerSuit = document.getElementById("cards-per-suit-input").value
        deckSizeEl.value = numSuits * cardsPerSuit
    }
})