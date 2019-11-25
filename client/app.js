import React, { Component } from 'react';
import 'bootstrap/dist/css/bootstrap.css';

// Reference: 
const loadImage = (input, imageRef) => {
  // input: target
  // imageRef: ref to <img /> tag.
  if (input.files && input.files[0]) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
      imageRef.current.src = e.target.result;
    }
    
    reader.readAsDataURL(input.files[0]);
  }
};

class App extends Component {
  constructor(props) {
    super(props);
    this.imageARef = React.createRef();
    this.imageBRef = React.createRef();
  }
  
  // Experimental class field syntax used to provide "this" context.
  // See this page: https://reactjs.org/docs/handling-events.html
  previewImageA = (e) => {
    loadImage(e.target, this.imageARef);
  }

  previewImageB = (e) => {
    console.log('preview B');
    loadImage(e.target, this.imageBRef);
  }
  
  render() {
    return (
      <div>
        <form>
          <div className="form-group">
            <label htmlFor="image_a">Image A</label>
            <input type="file" className="form-control-file" id="image_a" onChange={this.previewImageA} />
          </div>
        </form>

        <img ref={this.imageARef} src="" alt="image_a" />

        <form>
          <div className="form-group">
            <label htmlFor="image_b">Image B</label>
            <input type="file" className="form-control-file" id="image_b" onChange={this.previewImageB} />
          </div>
        </form>

        <img ref={this.imageBRef} src="" alt="image_b" />
      </div>
    );
  }
};

export default App;