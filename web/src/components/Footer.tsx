import { Icon } from "./Icon";
import "./Footer.css";

export function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer__inner">
          <p className="footer__note">
            A screening aid, not a structural assessment. Results should be confirmed by a qualified
            engineer.
          </p>
          <div className="footer__row">
            <p>© 2025 Kritish Dhital. All rights reserved.</p>
            <a
              className="link"
              href="https://github.com/KritishX/Crack-Analysis-using-AI"
              target="_blank"
              rel="noreferrer"
            >
              <Icon name="github" size={16} />
              Source on GitHub
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
