import React, { Component } from 'react';
import McganApp from './McganApp';

const run_state = {
  left: 'horse',
  right: 'zebra',
  runId: 'mask_horse2zebra_h128_nres=3_simpled',
}


export default class App extends Component {
  // state = {
    
  // }
  
  render() {
    const { left, right, runId } = run_state;
    return <McganApp left={left} right={right} runId={runId} />;
  }
}