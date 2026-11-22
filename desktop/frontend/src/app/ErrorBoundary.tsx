import { Component, type ErrorInfo, type ReactNode } from 'react';
import { ErrorState, Page } from '@/components/ui';

interface State {
  error: Error | null;
}

/** Empêche une erreur de rendu d'un module de faire tomber toute l'application. */
export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  override state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Erreur de rendu', error, info.componentStack);
  }

  override render() {
    if (this.state.error) {
      return (
        <Page>
          <ErrorState
            title="Cette page a rencontré une erreur"
            error={this.state.error.message}
            onRetry={() => this.setState({ error: null })}
          />
        </Page>
      );
    }
    return this.props.children;
  }
}
