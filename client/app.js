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

  handleSubmit = (e) => {
    e.preventDefault();

    const { realASrc } = this.state;
    postData('http://127.0.0.1:5000', { realASrc })
      .then(res => {
        res.json().then(json => {
          this.setState({
            response: json,
            fakeBSrc: json['fakeBSrc'],
          })
        });
        window.myResponse = res;
      }, rej => console.log(rej));
  }

  renderFakeB() {
    return (
      <div>
        <div>fakeB</div>
        <img src={this.state.fakeBSrc} alt="fakeB" />
      </div>
    );
  }

  renderResponseDebug() {
    return (
      <div>
        <br/>
        <br/>
        <br/>
        <div>
          <div>Response.json:</div>
          <div>{JSON.stringify(this.state.response)}</div>
        </div>
      </div>
    );
  }
  
  render() {
    const { realASrc, fakeBSrc, response } = this.state;

    return (
      <div className="container">
        <form onSubmit={this.handleSubmit}>
          <div className="form-group">
            <label htmlFor="image_a">Image A</label>
            <input type="file" className="form-control-file" id="image_a" onChange={this.previewImageA} />

            <img src={realASrc} alt="image_a" />
          </div>

          <button className="btn btn-primary" type="submit">Generate</button>
        </form>

        {fakeBSrc && this.renderFakeB()}
        {response && this.renderResponseDebug()}
      </div>
    );
  }
};

export default App;