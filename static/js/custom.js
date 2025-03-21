
let selectedModelWhite = null;
let selectedModelBlack = null;


fetch('/models')
.then(res => res.json())
.then(models => {
    const whiteSelector = document.getElementById('modelSelectorWhite');
    const blackSelector = document.getElementById('modelSelectorBlack');

    models.forEach(model => {
        const optionWhite = document.createElement('option');
        optionWhite.value = model;
        optionWhite.textContent = model;
        whiteSelector.appendChild(optionWhite);

        const optionBlack = document.createElement('option');
        optionBlack.value = model;
        optionBlack.textContent = model;
        blackSelector.appendChild(optionBlack);
    });

   
    whiteSelector.value = "user";
    blackSelector.value = "stockfish";
    selectedModelWhite = whiteSelector.value;
    selectedModelBlack = blackSelector.value;
});



function continueAIPlay() {
    const modelWhite = document.getElementById('modelSelectorWhite').value;
    const modelBlack = document.getElementById('modelSelectorBlack').value;

    
    if ((game.turn() === 'w' && modelWhite === 'user') ||
        (game.turn() === 'b' && modelBlack === 'user') ||
        game.game_over()) {
        return;
    }

    fetch('/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            from: "AI",
            to: "trigger",
            fen: game.fen(),
            model_white: modelWhite,
            model_black: modelBlack
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.eval !== undefined) {
            evalHistory.push(data.eval);
            moveLabels.push(evalHistory.length);
            evalChart.update();
        }
        if (data.fen) {
            game.load(data.fen);
            board.position(data.fen);
            updateStatus();

            
            setTimeout(continueAIPlay, 200);
        }
    });
}


document.getElementById('startAI').addEventListener('click', () => {
    continueAIPlay();
});

document.getElementById('resetBtn').addEventListener('click', () => {
    fetch('/reset', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        game.load(data.fen);
        board.position(data.fen);
        updateStatus();

        
        evalHistory.length = 0;
        moveLabels.length = 0;
        evalChart.update();
    });
});

document.getElementById('modelSelectorWhite').addEventListener('change', function () {
    selectedModelWhite = this.value;
});

document.getElementById('modelSelectorBlack').addEventListener('change', function () {
    selectedModelBlack = this.value;
});

const evalHistory = [];
const moveLabels = [];

const ctx = document.getElementById('evalChart').getContext('2d');
const evalChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: moveLabels,
        datasets: [{
            label: 'Stockfish Eval (centipawns)',
            data: evalHistory,
            fill: false,
            tension: 0.2
        }]
    },
    options: {
        scales: {
            y: {
                title: {
                    display: true,
                    text: 'Centipawn Score (0 = Equal)'
                }
            },
            x: {
                title: {
                    display: true,
                    text: 'Move Number'
                }
            }
        }
    }
});




var board = null;
var game = new Chess();

function onDragStart(source, piece, position, orientation) {
    if (game.game_over()) return false;
    if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
        (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
        return false;
    }
}


function onDrop(source, target) {
    var move = game.move({
        from: source,
        to: target,
        promotion: 'q'
    });

    if (move === null) return 'snapback';

    updateStatus();
    board.position(game.fen());

    const selectedModelWhite = document.getElementById('modelSelectorWhite').value;
    const selectedModelBlack = document.getElementById('modelSelectorBlack').value;


    fetch('/move', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            from: source,
            to: target,
            fen: game.fen(),
            model_white: selectedModelWhite,
            model_black: selectedModelBlack
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.eval !== undefined) {
            evalHistory.push(data.eval);
            moveLabels.push(evalHistory.length);
            evalChart.update();
        }
        if (data.fen) {
            game.load(data.fen);
            board.position(data.fen);
            updateStatus();
        }
    });
}



function updateStatus() {
    var status = '';
    if (game.in_checkmate()) {
        status = 'Game over, ' + (game.turn() === 'b' ? 'Black' : 'White') + ' is in checkmate.';
    } else if (game.in_draw()) {
        status = 'Game over, drawn position';
    } else {
        status = game.turn() === 'b' ? 'Black to move' : 'White to move';
        if (game.in_check()) {
            status += ', in check';
        }
    }
    console.log(status);
    document.getElementById('status').textContent = status;
}

var config = {
    draggable: true,
    position: 'start',
    pieceTheme: '/static/img/chesspieces/wikipedia/{piece}.png',
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: function () {
        board.position(game.fen());
    }
};

board = Chessboard('board', config);
updateStatus();
