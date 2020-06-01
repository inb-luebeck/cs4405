import numpy as np
import torch
from IPython.core.display import display, HTML

def make_prediction_canvas(canvas_size, prediction_func_name):
        canvas_html = '''
<div>
<p>Drawing canvas:</p>
<canvas id="canvas" width="''' + str(canvas_size[0]) + '''" height="''' + str(canvas_size[1]) + '''" style="border: 5px solid black"></canvas>
<button onclick="predict()">Predict</button>
<button onclick="clear_canvas()">Clear canvas</button>
<p id="predictionfield">Prediction:</p>
</div>
<script type="text/Javascript">
function prediction_callback(data){
    if (data.msg_type === 'execute_result') {
        document.getElementById("predictionfield").innerHTML = "Prediction: " + data.content.data['text/plain']
        /*$('#predictionfield').html("Prediction: " + data.content.data['text/plain'])*/
    } else {
        console.log(data)
    }
}
function predict(){
    var imgData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    imgData = Array.prototype.slice.call(imgData.data).filter(function (data, idx) { return idx % 4 == 3; })

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
        var command = "''' + prediction_func_name + '''(" + JSON.stringify(imgData) + ")"
        document.getElementById("predictionfield").innerHTML = "Prediction: calculating..."
        /*$('#predictionfield').html("Prediction: calculating...")*/

        var kernel = IPython.notebook.kernel;
        kernel.execute(command, {iopub: {output: prediction_callback}}, {silent: false});
    } else if (kernelAPI === "google") {
        google.colab.kernel.invokeFunction("''' + prediction_func_name + '''", [imgData], {})
        .then(function(result) {
            prediction_callback({msg_type: 'execute_result', content: {data: result.data}})
        })
    } else {
        console.error('no kernel api found to invoke predictions!')
    }
}

canvas = document.getElementById('canvas')
ctx = canvas.getContext("2d")

var clickX = new Array();
var clickY = new Array();
var clickDrag = new Array();
var paint;

function clear_canvas() {    
    clickX = new Array();
    clickY = new Array();
    clickDrag = new Array();
    
    redraw();
}

function addClick(x, y, dragging)
{
  clickX.push(x);
  clickY.push(y);
  clickDrag.push(dragging);
}

var canvas = document.getElementById("canvas")
/*$('#canvas').mousedown(*/
canvas.addEventListener('mousedown', function(e){
  var boundingRect = canvas.getBoundingClientRect()
  var mouseX = e.pageX - boundingRect.left;
  var mouseY = e.pageY - boundingRect.top;
  
  paint = true;
  addClick(mouseX, mouseY);
  redraw();
});

/*$('#canvas').mousemove(*/
canvas.addEventListener('mousemove', function(e){
  if(paint){
    var boundingRect = canvas.getBoundingClientRect()
    addClick(e.pageX - boundingRect.left, e.pageY - boundingRect.top, true);
    redraw();
  }
});

/*$('#canvas').mouseup(*/
canvas.addEventListener('mouseup', function(e){
  paint = false;
});

/*$('#canvas').mouseleave(*/
canvas.addEventListener('mouseleave', function(e){
  paint = false;
});

function redraw(){
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // Clears the canvas
  
  ctx.strokeStyle = '#000000';//"#df4b26";
  ctx.lineJoin = "round";
  ctx.lineWidth = 20;
  for(var i=0; i < clickX.length; i++) {
    ctx.beginPath();
    if(clickDrag[i] && i){
      ctx.moveTo(clickX[i-1], clickY[i-1]);
     }else{
       ctx.moveTo(clickX[i]-1, clickY[i]);
     }
     ctx.lineTo(clickX[i], clickY[i]);
     ctx.closePath();
     ctx.stroke();
  }
}
</script>
'''
        display(HTML(canvas_html)) 

import idx2numpy
from torch.utils.data import Dataset, DataLoader

class EMNIST(Dataset):
    def __init__(self, images_path, labels_path):
        self.images = idx2numpy.convert_from_file(images_path)
        self.labels = idx2numpy.convert_from_file(labels_path)
        
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def evaluate(epoch, model, criterion, test_loader, use_gpu):
    #test
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    batch_cnt = 0
    with torch.no_grad():
        model.eval()
        predictions = []
        labels = []
        for batch_idx, (x, target) in enumerate(test_loader):
            target = target.type(dtype=torch.long)
            x = x.type(dtype=torch.float)
            
            if use_gpu:
                x, target = x.cuda(), target.cuda()

            x.unsqueeze_(dim=1)#add channel dimension

            out = model(x)
            loss = criterion(out, target)
            pred_label = out.argmax(1)
            predictions.append(pred_label.cpu())
            labels.append(target.cpu())
            total_cnt += x.size()[0]
            correct_cnt += pred_label.eq(target).sum()
            ave_loss += loss.item()
            
            batch_cnt += 1

        #concatenate predictions from batches into one long list
        predictions = torch.cat(predictions, dim=0).numpy()
        #concatenate labels from batches into one long list
        labels = torch.cat(labels, dim=0).numpy()
        confusmat = confusion_matrix(labels, predictions)
        
        
    print('==>>> epoch: {}, test loss: {:.6f}, test acc: {:.4f}'.format(
            epoch, ave_loss / batch_cnt, correct_cnt.item() * 1.0 / total_cnt))
    
    plt.matshow(confusmat)
    #plt.matshow(confusion_matrix[10:,10:].numpy())
    plt.show()

def train(epoch, model, loss_func, optimizer, train_loader, use_gpu):
  #put model into training mode
  model.train()
  #iterate over batches for one epoch
  correct_cnt = 0
  total_cnt = 0
  for batch_idx, (x, target) in enumerate(train_loader):
      #convert to the correct data types
      target = target.type(dtype=torch.long)
      #convert images from byte-valued pixels to float-valued pixels
      x = x.type(dtype=torch.float)
      
      if use_gpu:
          x, target = x.cuda(), target.cuda()
      
      #Add the singular color channel dimension to transform the dimensions from (batch_size, 28, 28) to (batch_size, 1, 28, 28) as expected by the Conv2d module
      x.unsqueeze_(dim=1)
      
      #Reset gradients to zero
      optimizer.zero_grad()
      
      #predict
      out = model(x)
      #calculate loss of the prediction
      loss = loss_func(out, target)
      #perform back propagation
      loss.backward()
      
      #let the optimizer adjust the weights according to the gradients
      optimizer.step()

      pred_label = out.argmax(1)
      correct_cnt += pred_label.eq(target).sum()
      total_cnt += x.size()[0]
      
      #periodically give some feedback
      if (batch_idx + 1) % 500 == 0:
          #calculate the prediction accuracy on the batch
          accuracy_on_batches_window = float(correct_cnt.item()) / total_cnt
          print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}, train acc: {:.4f}'.format(
                  epoch, batch_idx+1, loss.item(), accuracy_on_batches_window))