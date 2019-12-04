import React, { Component } from 'react';
import McganApp from './McganApp';

const run_state = {
  left: 'horse',
  right: 'zebra',
}


export default class App extends Component {
  // state = {
    
  // }
  
  render() {
    const { left, right } = run_state;
    return <McganApp left={left} right={right} />;
  }
}