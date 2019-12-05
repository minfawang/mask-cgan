import React, { Component } from 'react';

const style = {
  width: '128px',
  height: '128px',
  border: '1px solid black',
};

export default class Mask extends Component {
  canvasRef = React.createRef();

  maybeDrawMask = () => {
    const { percent } = this.props;
    if (percent < 0) { return; }
    
    const canvas = this.canvasRef.current;
    const ctx = canvas.getContext('2d');

    const { width, height } = canvas;

    // Fill context region in white.
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);

    // Fill mask region in black.
    ctx.fillStyle = 'black';
    const mid_w = Math.floor(width / 2);
    const mid_h = Math.floor(height / 2);
    ctx.fillRect(mid_w * (1 - percent), mid_h * (1 - percent),
                 width * percent, height * percent);
  }

  componentDidMount() {
    this.maybeDrawMask();
  }

  componentDidUpdate() {
    this.maybeDrawMask();
  }
  
  renderMaskFromSrc() {
    const { src } = this.props;
    if (!src) {
      return <div>Generating ...</div>;
    }

    return <img src={src} />;
  }
  
  render() {
    const { percent } = this.props;

    if (percent < 0) {
      return this.renderMaskFromSrc();
    }

    return (
        <canvas ref={this.canvasRef} style={style}></canvas>
    );
  }
}