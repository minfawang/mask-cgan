import React, { Component } from 'react';
import 'bootstrap/dist/css/bootstrap.css';
import {DEFAULT_REAL_A_SRC} from './consts';
import Mask from './Mask';

function object2UrlParams(params) {
  return Object.keys(params).map((key) => {
      return encodeURIComponent(key) + '=' + encodeURIComponent(params[key])
  }).join('&');
}


function maskVal2Percent(val) {
  return {
    0: 0.5,
    1: 0.8,
    2: 1.0,
    3: -1,  // Random: I feel lucky.
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

class McganApp extends Component {
  state = {
    imgSrcs: {},  // { fileName: dataUrl }
    // realSrc: DEFAULT_REAL_A_SRC,
    realSrc: null,
    response: '',
    fakeSrc: '',
    maskValue: 2,
    maskSrc: '',  // If maskValue == 3 (I feel lucky), this mask may become the dataURL of the mask returned from server.
    isA2B: true,
    loading: false,
    showDebug: false,
  };
  
  // Experimental class field syntax used to provide "this" context.
  // See this page: https://reactjs.org/docs/handling-events.html
  previewImage = (e) => {
    const files = event.target.files;
    console.log('this.loadImage()');
    // input: target
    // imageRef: ref to <img /> tag.
    if (files && files[0]) {
      const reader = new FileReader();
      
      reader.onload = e => {
        console.log('reader.onload()');
        const src = e.target.result;
        this.setState({ realSrc: src, realSrcKey: null });
      }
      
      reader.readAsDataURL(files[0]);
    }
  }

  updateRealSrcKey = (e) => {
    const realSrcKey = e.currentTarget.id;
    const realSrc = this.state.imgSrcs[realSrcKey];
    this.setState({ realSrcKey, realSrc }, this.generateImage);
  }

  generateImage = () => {
    const { realSrc, maskValue, isA2B } = this.state;
    const { runId } = this.props;

    const maskPercent = maskVal2Percent(maskValue);
    const url = `/generate`;
    console.log('posting to: ', url);
    postData(url, { runId, isA2B, realSrc, maskPercent })
      .then(res => {
        res.json().then(json => {
          this.setState({
            response: json,
            loading: false,
            fakeSrc: json['fakeSrc'],
            maskSrc: json['maskSrc'],
          })
        });
        window.myResponse = res;
      }, rej => console.log(rej));
    this.setState({ loading: true });
  }

  handleSubmit = (e) => {
    e.preventDefault();
    this.generateImage();
  }

  handleMaskChange = e => {
    const maskValue = e.target.value;
    this.setState({ maskValue }, this.generateImage);
  }

  toggleShowDebug = () => {
    const { showDebug } = this.state;
    this.setState({ showDebug: !showDebug });
  }

  renderFakeImage() {
    const { isA2B, loading } = this.state;
    
    return (
      <div className="col-md-4">
        <div style={{ minHeight: '70px' }}>fake {isA2B ? 'Zebra' : 'Horse'} {loading && 'generating ...'}</div>
        <img src={this.state.fakeSrc} alt="fake_image" />
      </div>
    );
  }

  renderResponseDebug() {
    const { showDebug } = this.state;
    const responseView = (
      <div>
        <div>Response.json:</div>
        <div>{JSON.stringify(this.state.response)}</div>
      </div>
    );
    
    return (
      <div style={{ overflowWrap: 'break-word' }}>
        <br/>
        <br/>
        <br/>
        <button className="btn btn-light" onClick={this.toggleShowDebug}>
          {showDebug ? 'Hide' : 'Show'} Debug
        </button>
        {showDebug && responseView}
      </div>
    );
  }

  setA2B = () => { this.setState({ isA2B: true }, this.fetchImgs); }
  setB2A = () => { this.setState({ isA2B: false }, this.fetchImgs); }

  clickLucky = () => {
    this.setState({ maskValue: 3 });
  }

  fetchImgs = (e) => {
    if (!!e) {
      e.preventDefault();
    }
    
    const { left, right } = this.props;
    const { isA2B } = this.state;
    const side = isA2B ? 'A' : 'B';
    const url = `/rand_imgs/${left}2${right}/${side}`;
    
    postData(url)
      .then(res => {
        res.json().then(json => {
          const { imgSrcs } = json;
          this.setState({ imgSrcs, realSrc: null, realSrcKey: null, response: json });
        });
      }, rej => console.log(rej));
  }

  //////////////////////////////////////////////////////////////
  // Lifecycle methods
  componentDidMount() {
    this.fetchImgs();
  }
  
  //////////////////////////////////////////////////////////////
  // Render methods
  
  renderA2BRadio() {
    const { isA2B } = this.state;
    const { left, right } = this.props;
    const a2b_label = `${left}2${right}`;
    const b2a_label = `${right}2${left}`;
    
    return (
      <React.Fragment>
        <div className="custom-control custom-radio custom-control-inline">
          <input type="radio" id="customRadioInline1" name="customRadioInline1" className="custom-control-input" checked={isA2B} onChange={this.setA2B} />
          <label className="custom-control-label" htmlFor="customRadioInline1">{a2b_label}</label>
        </div>
        <div className="custom-control custom-radio custom-control-inline">
          <input type="radio" id="customRadioInline2" name="customRadioInline1" className="custom-control-input" checked={!isA2B} onChange={this.setB2A} />
          <label className="custom-control-label" htmlFor="customRadioInline2">{b2a_label}</label>
        </div>
      </React.Fragment>
    );
  }

  renderMaskRange() {
    const { maskValue } = this.state;
    const percent = maskVal2Percent(maskValue);
    const percentStr = `${percent * 100} %`;

    return (
      <div className="form-group col-md-4">
        <label htmlFor="formControlRange">Mask percent: {percentStr}</label>
        <input type="range" className="form-control-range custom-range" min={0} max={2} id="formControlRange" onChange={this.handleMaskChange} value={maskValue} />
        <Mask percent={percent} />
      </div>
    );
  }

  renderMaskRadio() {
    const { maskValue, maskSrc } = this.state;
    const percent = maskVal2Percent(maskValue);
    return (
      <div className="form-group col-md-4">
        <label htmlFor="formControlRange">Mask percent</label>

        <div>
          <div className="form-check form-check-inline">
            <input className="form-check-input" type="radio" id="inlineradio1" value="0" checked={maskValue == '0'} onChange={this.handleMaskChange} />
            <label className="form-check-label" htmlFor="inlineradio1">50%</label>
          </div>
          <div className="form-check form-check-inline">
            <input className="form-check-input" type="radio" id="inlineradio2" value="1" checked={maskValue == '1'} onChange={this.handleMaskChange} />
            <label className="form-check-label" htmlFor="inlineradio2">80%</label>
          </div>
          <div className="form-check form-check-inline">
            <input className="form-check-input" type="radio" id="inlineradio3" value="2" checked={maskValue == '2'} onChange={this.handleMaskChange} />
            <label className="form-check-label" htmlFor="inlineCheckbox3">100%</label>
          </div>
          <div>
            <button className="btn btn-outline-success" onClick={this.clickLucky}>I feel lucky</button>
          </div>
        </div>

        <Mask percent={percent} src={maskSrc} />
      </div>
    );
  }
  
  renderRealImage() {
    const { realSrc, realSrcKey, imgSrcs, isA2B }  = this.state;

    const randImgs = Object.keys(imgSrcs).map(key => {
      const imgSrc = imgSrcs[key];
      let style = {maxWidth: '128px'};
      if (realSrcKey == key) {
        style = { ...style, borderStyle: 'dashed', borderColor: 'red' };
      }
      
      return <img src={imgSrc}
                  id={key}
                  key={key}
                  alt="image"
                  style={style}
                  onClick={this.updateRealSrcKey}
              />;
    });
    
    const uploadedImg = (
      <img src={realSrc} alt="image" style={{maxWidth: '128px'}} />
    );
    
    return (
      <div className="form-group col-md-4">
        <label htmlFor="image">{isA2B ? 'Horse' : 'Zebra'} Image</label>
        <input type="file" className="form-control-file" id="image" onChange={this.previewImage} />
        {(!!realSrc && !realSrcKey) ? uploadedImg : randImgs}
        <div>
          <button className="btn btn-outline-success" onClick={this.fetchImgs}>Fetch random images</button>
        </div>
      </div>
    );
  }
  
  render() {
    const { realSrc, fakeSrc, loading } = this.state;
    const { left, right } = this.props;
    const title = `${left} vs. ${right}`;

    return (
      <div className="container">
        <h1>{title}</h1>
        <form onSubmit={this.handleSubmit}>
          {this.renderA2BRadio()}

          <br/>
          <br/>

          <div className="row">
            {this.renderRealImage()}
            {this.renderMaskRadio()}

            {fakeSrc && this.renderFakeImage()}
          </div>
        </form>

        {this.renderResponseDebug()}
      </div>
    );
  }
};

export default McganApp;