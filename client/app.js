import React, { Component } from 'react';
import McganApp from './McganApp';
import Dropdown from 'react-bootstrap/Dropdown';
import DropdownButton from 'react-bootstrap/DropdownButton'

const RUN_INFOS = [
  // run_id, left, right
  ['mask_horse2zebra_h128_nres=3_simpled', 'horse', 'zebra'],
  ['mask_monet2photo_h128_nres=3_simpled', 'monet', 'photo'],
  ['mask_vangogh2photo_h128_nres=3_simpled', 'vangogh', 'photo'],
  ['p80mask_vangogh2photo_h128_nres=3_simpled', 'vangogh', 'photo'],
]


const run_state = {
  left: 'horse',
  right: 'zebra',
  runId: 'mask_horse2zebra_h128_nres=3_simpled',
}


function findRunInfo(runId) {
  for (let i = 0; i < RUN_INFOS.length; ++i) {
    const runInfo = RUN_INFOS[i];
    if (runInfo[0] == runId) {
      return runInfo;
    }
  }
}


export default class App extends Component {
  state = {
    runId: RUN_INFOS[0][0],
  }

  switchRun = (runId, e) => {
    this.setState({ runId });
  }

  renderRunDropdown() {
    const dropdownStyle = { display: 'inline-block' };
    
    const runItems = RUN_INFOS.map(runInfo => {
      const [runId, left, right] = runInfo;
      const active = (this.state.runId == runId);
      return <Dropdown.Item key={runId} onSelect={this.switchRun} active={active} eventKey={runId}>{runId}</Dropdown.Item>;
    });
    
    return (
      <div>
        <h3>Choose Model</h3>
        <DropdownButton id="dropdown-basic-button" title={this.state.runId} style={dropdownStyle}>
        {runItems}
      </DropdownButton>
      </div>
    );
  }
  
  render() {
    const { runId } = this.state;
    const [_, left, right] = findRunInfo(runId);

    const instructionStyle = {
      border: '2px',
      borderStyle: 'solid',
      color: '#888',
    };
    
    return (
      <div className="container">
        <h1>Mask CycleGAN Demo</h1>
        <div><a href="https://www.github.com/minfawang/mask-cgan">Source on Github</a></div>

        <ul style={instructionStyle}>
          <b>Instructions:</b>
          <li>(optional) Choose a model</li>
          <li>(optional) Choose a direction</li>
          <li>Choose an image</li>
          <li>(optional) Choose a mask</li>
        </ul>
        
        {this.renderRunDropdown()}
        <br />
        <McganApp key={runId} left={left} right={right} runId={runId} />
      </div>
    );
  }
}