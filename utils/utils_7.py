from IPython.core.display import display, HTML

def make_prediction_field(dictionary, prediction_func_name):
    input_text_html = '''
<div>
<p id="currentwords"></p>
<input type="text" list="words" id="textfield" autocomplete=off>
<button onclick="predict()">Predict</button>
<datalist id="words">
</datalist>
<p id="predictionfield"></p>
</div>
<script type="text/Javascript">
var dictionary = ''' + repr(dictionary) + '''

var textfield = document.getElementById('textfield')

var inputwords = []

/*$("#textfield").on('propertychange change click keyup input paste', function(evt){*/
function textFieldEventHandler(evt) {
    
    /*var newwords = $(this).val().split(" ")*/
    var newwords = textfield.value.split(" ")
    if (newwords.length > 0) {
        var remaining = newwords.pop() //the last one is still being worked on
        newwords = newwords.filter(function(word) {return dictionary[word] != undefined})
        
        inputwords = inputwords.concat(newwords)
        
        if (remaining.length == 0) {
            document.getElementById("words").innerHTML = ""
            /*$("#words").html("")*/
        } else {
            while(remaining.length > 0) {
                var matchingwords = Object.keys(dictionary).filter(function(word) {return word.startsWith(remaining)})
                if (matchingwords.length > 0) {
                    break;
                }
                remaining = remaining.substr(0, remaining.length-1) //try again without the last character
            }
            //too many suggestions bring the browser to its knees
            if (matchingwords.length > 20) {
                matchingwords = []
            }
            var suggestions = matchingwords.map(function(word) {return '<option>' + word + '</option>'}).join("")
            document.getElementById("words").innerHTML = suggestions
            /*$("#words").html(suggestions)*/
        }
        
        document.getElementById("currentwords").textContent = inputwords.join(" ")
        /*$("#currentwords").text(inputwords.join(" "))*/
        textfield.value = remaining
        /*$(this).val(remaining)*/
    }
    
    if (textfield.value.length/*$(this).val().length*/ == 0 && evt.type == 'keyup' && evt.key == 'Backspace' && inputwords.length > 0) {
        //remove previous word
        inputwords.pop()
        
        document.getElementById("currentwords").textContent = inputwords.join(" ")
        /*$("#currentwords").text(inputwords.join(" "))*/
    }
}

textfield.addEventListener('propertychange', textFieldEventHandler)
textfield.addEventListener('change', textFieldEventHandler)
textfield.addEventListener('click', textFieldEventHandler)
textfield.addEventListener('keyup', textFieldEventHandler)
textfield.addEventListener('input', textFieldEventHandler)
textfield.addEventListener('paste', textFieldEventHandler)

function prediction_callback(data) {
    if (data.msg_type === 'execute_result') {
        var sentence = data.content.data['text/plain']
        sentence = sentence.substring(1, sentence.length - 1)
        inputwords = sentence.split(" ").filter(function(word) {return word != '<eos>'})
        document.getElementById("currentwords").textContent = inputwords.join(" ")
        /*$("#currentwords").text(inputwords.join(" "))*/
    } else {
        console.log(data)
    }
    document.getElementById("predictionfield").textContent = ""
    /*$("#predictionfield").text("")*/
}

function predict() {
    var kernelAPI = undefined
    try {
      //check if defined
      if (IPython) {
        kernelAPI = "IPython"
      }
    } catch(err) {
    }

    try {
      //check if defined
      if (google) {
        kernelAPI = "google"
      }
    } catch(err) {
    }

    if (kernelAPI === "IPython") {
        var command = "''' + prediction_func_name + '''(" + JSON.stringify(inputwords) + ")"
        document.getElementById("predictionfield").textContent = "predicting..."
        /*$('#predictionfield').html("Prediction: calculating...")*/

        var kernel = IPython.notebook.kernel;
        kernel.execute(command, {iopub: {output: prediction_callback}}, {silent: false});
    } else if (kernelAPI === "google") {
        document.getElementById("predictionfield").textContent = "predicting..."

        google.colab.kernel.invokeFunction("''' + prediction_func_name + '''", [inputwords], {})
        .then(function(result) {
            prediction_callback({msg_type: 'execute_result', content: {data: result.data}})
        })
    } else {
        console.error('no kernel api found to invoke predictions!')
    }
}
</script>
'''

    display(HTML(input_text_html)) 


import torch
from torch.utils.data import Dataset

class PTB(Dataset):
    def __init__(self, filename):
        with open(filename, "r") as file:
            data = file.readlines()
            data = [sentence.split(" ") for sentence in data]
            
            word_set = set()
            
            for sentence in data:
                sentence.pop(0)#remove the empty start-marker
                sentence[-1] = "<eos>"#replace linebreak with end-of-sentence word
                for word in sentence:
                    word_set.add(word)
                    
            self.word_set = word_set
            self.sentences = data    
            #print(len(data))
            #example = data[10]
            #print(example)
            #print(word_set)
            #self.sentences = 
            
    def encode_sentences(self, dictionary):
        self.sentences_encoded = [[dictionary[word] for word in sentence] for sentence in self.sentences]
        
    def __len__(self):
        return len(self.sentences_encoded)
    
    def __getitem__(self, index):
        return torch.tensor(self.sentences_encoded[index])


class TrainedModel(torch.nn.Module):
    def __init__(self, dict_size):
        super(TrainedModel, self).__init__()
        
        embedding_dim = 1000#650#320#650#32
        hidden_size = 1000#650#320#650#48
        
        self.embedding = torch.nn.Embedding(dict_size, embedding_dim)
        #self.rnn = torch.nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.rnn = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.f = torch.nn.Linear(hidden_size, dict_size)
        
    def forward(self, batch, lengths):
        x = self.embedding(batch)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, torch.tensor(lengths), batch_first=True)
        
        outputs, hidden = self.rnn(x)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        x = self.f(outputs)
        
        #x = self.f(self.dropout(outputs.data))
        #x = torch.nn.utils.rnn.PackedSequence(x, batch_sizes=outputs.batch_sizes)
        #x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        ##x = F.softmax(self.f(outputs), dim=-1)
        
        return x

def get_trained_model():
    trained_state_dict, original_dictionary = torch.load("data/trained_model_weights.pt")
    original_inv_dictionary = {v: k for k, v in original_dictionary.items()}
    trained_model = TrainedModel(len(original_dictionary))
    trained_model.load_state_dict(trained_state_dict)

    return trained_model, original_dictionary, original_inv_dictionary