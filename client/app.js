import React, { Component } from 'react';
import 'bootstrap/dist/css/bootstrap.css';
import {DEFAULT_REAL_A_SRC} from './consts';
import Mask from './mask';

function object2UrlParams(params) {
  return Object.keys(params).map((key) => {
      return encodeURIComponent(key) + '=' + encodeURIComponent(params[key])
  }).join('&');
}


function maskVal2Percent(val) {
  return {
    0: 0.5,
    1: 0.8,
    2: 1.0
  }[val];
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
    realSrc: DEFAULT_REAL_A_SRC,
    response: '',
    fakeSrc: '',
    maskValue: 2,
    isA2B: true,
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
  previewImage = (e) => {
    this.loadImage(e, 'realSrc');
  }

  generateImage = () => {
    const { realSrc, maskValue, isA2B } = this.state;
    const maskPercent = maskVal2Percent(maskValue);
    postData('http://localhost:5000/generate', { isA2B, realSrc, maskPercent })
      .then(res => {
        res.json().then(json => {
          this.setState({
            response: json,
            fakeSrc: json['fakeSrc'],
          })
        });
        window.myResponse = res;
      }, rej => console.log(rej));
  }

  handleSubmit = (e) => {
    e.preventDefault();
    this.generateImage();
  }

  handleMaskChange = e => {
    const maskValue = e.target.value;
    this.setState({ maskValue }, this.generateImage);
  }

  renderFakeImage() {
    const { isA2B } = this.state;
    
    return (
      <div className="col-md-4">
        <div style={{ minHeight: '70px' }}>fake {isA2B ? 'Zebra' : 'Horse'}</div>
        <img src={this.state.fakeSrc} alt="fake_image" />
      </div>
    );
  }

  renderResponseDebug() {
    return (
      <div style={{ overflowWrap: 'break-word' }}>
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

  setA2B = () => { this.setState({ isA2B: true }, this.generateImage); }
  setB2A = () => { this.setState({ isA2B: false }, this.generateImage); }

  renderA2BRadio() {
    const { isA2B } = this.state;
    return (
      <React.Fragment>
        <div className="custom-control custom-radio custom-control-inline">
          <input type="radio" id="customRadioInline1" name="customRadioInline1" className="custom-control-input" checked={isA2B} onChange={this.setA2B} />
          <label className="custom-control-label" htmlFor="customRadioInline1">Horse2Zebra</label>
        </div>
        <div className="custom-control custom-radio custom-control-inline">
          <input type="radio" id="customRadioInline2" name="customRadioInline1" className="custom-control-input" checked={!isA2B} onChange={this.setB2A} />
          <label className="custom-control-label" htmlFor="customRadioInline2">Zebra2Horse</label>
        </div>
      </React.Fragment>
    );
  }
  
  render() {
    const { realSrc, fakeSrc, response, maskValue, isA2B } = this.state;
    const percent = maskVal2Percent(maskValue);
    const percentStr = `${percent * 100} %`;

    return (
      <div className="container">
        <h1>Horse vs. Zebra</h1>
        <form onSubmit={this.handleSubmit}>
          {this.renderA2BRadio()}

          <br/>
          <br/>

          <div className="row">
            <div className="form-group col-md-4">
              <label htmlFor="image">{isA2B ? 'Horse' : 'Zebra'} Image</label>
              <input type="file" className="form-control-file" id="image" onChange={this.previewImage} />
              <img src={realSrc} alt="image" style={{maxWidth: '128px'}} />
            </div>

            <div className="form-group col-md-4">
              <label htmlFor="formControlRange">Mask percent: {percentStr}</label>
              <input type="range" className="form-control-range custom-range" min={0} max={2} id="formControlRange" onChange={this.handleMaskChange} value={maskValue} />
              <Mask percent={percent} />
            </div>

            {fakeSrc && this.renderFakeImage()}
          </div>
          

          <button className="btn btn-primary" type="submit">Generate</button>
        </form>

        {response && this.renderResponseDebug()}
      </div>
    );
  }
};

export default App;