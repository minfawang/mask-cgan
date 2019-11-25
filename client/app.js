import React, { Component } from 'react';
import 'bootstrap/dist/css/bootstrap.css';
import {DEFAULT_REAL_A_SRC} from './consts';

function object2UrlParams(params) {
  return Object.keys(params).map((key) => {
      return encodeURIComponent(key) + '=' + encodeURIComponent(params[key])
  }).join('&');
}

// Adapted from: https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
function postData(url = '', data = {}) {
  // Default options are marked with *
  return fetch(url, {
    method: 'POST', // *GET, POST, PUT, DELETE, etc.
    // mode: 'cors', // no-cors, *cors, same-origin
    // cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
    // credentials: 'same-origin', // include, *same-origin, omit
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
    },
    redirect: 'follow', // manual, *follow, error
    referrer: 'no-referrer', // no-referrer, *client
    body: JSON.stringify(data) // body data type must match "Content-Type" header
  });
}

class App extends Component {
  state = {
    realASrc: DEFAULT_REAL_A_SRC,
    realBSrc: '',
    response: '',
    fakeBSrc: '',
  };
  
  loadImage = (event, stateKey) => {
    const files = event.target.files;
    console.log('this.loadImage()');
    // input: target
    // imageRef: ref to <img /> tag.
    if (files && files[0]) {
      const reader = new FileReader();
      
      reader.onload = e => {
        console.log('reader.onload()');
        const src = e.target.result;
        this.setState({ [stateKey]: src });
      }
      
      reader.readAsDataURL(files[0]);
    }
  };
  
  // Experimental class field syntax used to provide "this" context.
  // See this page: https://reactjs.org/docs/handling-events.html
  previewImageA = (e) => {
    console.log('preview A');
    this.loadImage(e, 'realASrc');
  }

  previewImageB = (e) => {
    console.log('preview B');
    this.loadImage(e, 'realBSrc');
  }

  handleSubmit = (e) => {
    e.preventDefault();

    const { realASrc, realBSrc } = this.state;
    postData('http://127.0.0.1:5000', { realASrc, realBSrc })
      .then(res => {
        console.log(res);
        res.json().then(json => {
          this.setState({
            response: json,
            fakeBSrc: json['fakeBSrc'],
          })
        });
        window.myResponse = res;
      }, rej => console.log(rej));
  }
  
  render() {
    return (
      <div>
        <form onSubmit={this.handleSubmit}>
          <div className="form-group">
            <label htmlFor="image_a">Image A</label>
            <input type="file" className="form-control-file" id="image_a" onChange={this.previewImageA} />
          </div>
          <div className="form-group">
            <label htmlFor="image_b">Image B</label>
            <input type="file" className="form-control-file" id="image_b" onChange={this.previewImageB} />
          </div>

          <button type="submit">Submit</button>
        </form>

        <img src={this.state.realASrc} alt="image_a" />
        <img src={this.state.realBSrc} alt="image_b" />

        <div>
          <div>fakeB</div>
          <img src={this.state.fakeBSrc} alt="fakeB" />
        </div>

        <br/>
        <br/>
        <br/>
        <div>{JSON.stringify(this.state.response)}</div>
      </div>
    );
  }
};

export default App;